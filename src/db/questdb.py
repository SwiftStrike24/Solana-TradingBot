import logging
from datetime import datetime
import psycopg
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RPCMetrics:
    timestamp: datetime
    rpc_type: str  # 'helius' or 'jito'
    latency_ms: float
    success: bool
    tx_signature: Optional[str]
    route_count: int
    slippage_bps: Optional[int]
    compute_units: Optional[int]
    priority_fee: Optional[int]
    final_slippage_bps: Optional[int]
    total_fee_usd: Optional[float]  # Total transaction fees in USD
    swap_usd_value: Optional[float]  # Value of the swap in USD
    retry_count: int = 0
    slippage_adjustment: float = 0.0

class QuestDBClient:
    def __init__(self):
        # QuestDB PostgreSQL wire protocol connection settings
        self.connection_params = {
            "host": "localhost",
            "port": 8812,
            "user": "admin",
            "password": "quest",
            "dbname": "qdb"
        }
        
        self._init_tables()

    def _init_tables(self):
        """Initialize QuestDB tables for performance tracking if they don't exist."""
        try:
            with psycopg.connect(**self.connection_params) as conn:
                with conn.cursor() as cur:
                    # Create RPC performance metrics table if it doesn't exist
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS rpc_metrics (
                            timestamp TIMESTAMP,
                            rpc_type SYMBOL,
                            latency_ms DOUBLE,
                            success BOOLEAN,
                            tx_signature SYMBOL,
                            route_count INT,
                            slippage_bps INT,
                            compute_units INT,
                            priority_fee INT,
                            final_slippage_bps INT,
                            total_fee_usd DOUBLE,
                            swap_usd_value DOUBLE
                        ) timestamp(timestamp) PARTITION BY DAY;
                    """)
                    conn.commit()
                    
                    # Create hourly summary table if it doesn't exist
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS rpc_performance_hourly (
                            timestamp TIMESTAMP,
                            rpc_type SYMBOL,
                            avg_latency_ms DOUBLE,
                            success_rate DOUBLE,
                            total_txs LONG,
                            avg_compute_units DOUBLE,
                            avg_priority_fee DOUBLE,
                            avg_swap_value DOUBLE
                        ) timestamp(timestamp) PARTITION BY DAY;
                    """)
                    conn.commit()
                    logger.info("QuestDB tables initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing QuestDB tables: {str(e)}")
            raise

    def record_rpc_metrics(self, metrics: RPCMetrics):
        """Record RPC performance metrics."""
        try:
            with psycopg.connect(**self.connection_params) as conn:
                with conn.cursor() as cur:
                    # Debug logging
                    logger.info(f"""
{'='*80}
🔄 Recording RPC Metrics:

💰 Transaction Details:
   • Fee (USD): ${metrics.total_fee_usd:.4f} ({type(metrics.total_fee_usd)})
   • Swap Value: ${metrics.swap_usd_value:.2f} ({type(metrics.swap_usd_value)})
   • RPC: {metrics.rpc_type.upper()}
   • Routes: {metrics.route_count}

⚡ Performance:
   • Latency: {metrics.latency_ms:.2f}ms
   • Compute Units: {metrics.compute_units or 'N/A'}
   • Priority Fee: {metrics.priority_fee/1e9:.6f} SOL

🎯 Slippage:
   • Initial: {metrics.slippage_bps/100 if metrics.slippage_bps else 'N/A'}%
   • Final: {metrics.final_slippage_bps/100 if metrics.final_slippage_bps else 'N/A'}%
{'='*80}
""")
                    
                    # Ensure values are float with proper decimal precision
                    total_fee_usd = float(metrics.total_fee_usd) if metrics.total_fee_usd is not None else 0.0
                    swap_usd_value = float(metrics.swap_usd_value) if metrics.swap_usd_value is not None else 0.0
                    
                    cur.execute("""
                        INSERT INTO rpc_metrics (
                            timestamp,
                            rpc_type,
                            latency_ms,
                            success,
                            tx_signature,
                            route_count,
                            slippage_bps,
                            compute_units,
                            priority_fee,
                            final_slippage_bps,
                            total_fee_usd,
                            swap_usd_value
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::double precision, %s::double precision)
                    """, (
                        metrics.timestamp,
                        metrics.rpc_type,
                        metrics.latency_ms,
                        metrics.success,
                        metrics.tx_signature,
                        metrics.route_count,
                        metrics.slippage_bps,
                        metrics.compute_units,
                        metrics.priority_fee,
                        metrics.final_slippage_bps,
                        total_fee_usd,
                        swap_usd_value
                    ))
                    conn.commit()
                    logger.info(f"""
{'='*80}
✅ Metrics Successfully Recorded!

📊 Summary:
   • Fee: ${total_fee_usd:.4f}
   • Value: ${swap_usd_value:.2f}
   • Signature: {metrics.tx_signature[:8]}...{metrics.tx_signature[-8:] if metrics.tx_signature else 'N/A'}
{'='*80}
""")
        except Exception as e:
            logger.error(f"Error recording RPC metrics: {str(e)}")

    def get_rpc_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get RPC performance summary for the last N hours."""
        try:
            with psycopg.connect(**self.connection_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            rpc_type,
                            avg(latency_ms) as avg_latency,
                            sum(case when success then 1 else 0 end)::float / count(*)::float as success_rate,
                            count(*) as total_txs,
                            avg(compute_units) as avg_compute_units,
                            avg(priority_fee) as avg_priority_fee
                        FROM rpc_metrics
                        WHERE timestamp >= dateadd('h', -$1, now())
                        GROUP BY rpc_type
                    """, (hours,))
                    
                    results = cur.fetchall()
                    summary = {}
                    for row in results:
                        summary[row[0]] = {
                            'avg_latency_ms': row[1],
                            'success_rate': row[2],
                            'total_txs': row[3],
                            'avg_compute_units': row[4],
                            'avg_priority_fee': row[5]
                        }
                    return summary
        except Exception as e:
            logger.error(f"Error getting RPC performance summary: {str(e)}")
            return {}

    def get_latency_percentiles(self, rpc_type: str, hours: int = 24) -> Dict[str, float]:
        """Get latency percentiles for specific RPC."""
        try:
            with psycopg.connect(**self.connection_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            percentile_disc(0.50) WITHIN GROUP (ORDER BY latency_ms) as p50,
                            percentile_disc(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95,
                            percentile_disc(0.99) WITHIN GROUP (ORDER BY latency_ms) as p99
                        FROM rpc_metrics
                        WHERE rpc_type = $1
                        AND timestamp >= dateadd('h', -$2, now())
                    """, (rpc_type, hours))
                    
                    row = cur.fetchone()
                    if row:
                        return {
                            'p50': row[0],
                            'p95': row[1],
                            'p99': row[2]
                        }
                    return {}
        except Exception as e:
            logger.error(f"Error getting latency percentiles: {str(e)}")
            return {}

    def cleanup_old_data(self, days: int = 30):
        """Clean up data older than specified days."""
        try:
            with psycopg.connect(**self.connection_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM rpc_metrics 
                        WHERE timestamp < dateadd('d', -$1, now())
                    """, (days,))
                conn.commit()
                logger.info(f"Cleaned up data older than {days} days")
        except Exception as e:
            logger.error(f"Error cleaning up old data: {str(e)}")

    def _format_timestamp(self, ts: datetime) -> str:
        """Convert UTC timestamp to 12-hour format"""
        return ts.strftime('%Y-%m-%d %I:%M:%S %p')

    def get_formatted_metrics(self, hours: int = 24) -> list:
        """Get metrics with formatted timestamps"""
        try:
            with psycopg.connect(**self.connection_params) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            timestamp,
                            rpc_type,
                            latency_ms,
                            success,
                            tx_signature,
                            route_count,
                            slippage_bps,
                            compute_units,
                            priority_fee
                        FROM rpc_metrics
                        WHERE timestamp >= dateadd('h', -$1, now())
                        ORDER BY timestamp DESC
                    """, (hours,))
                    
                    results = []
                    for row in cur.fetchall():
                        results.append({
                            'human_time': self._format_timestamp(row[0]),
                            'timestamp': row[0],
                            'rpc_type': row[1],
                            'latency_ms': row[2],
                            'success': row[3],
                            'tx_signature': row[4],
                            'route_count': row[5],
                            'slippage_bps': row[6],
                            'compute_units': row[7],
                            'priority_fee': row[8]
                        })
                    return results
        except Exception as e:
            logger.error(f"Error getting formatted metrics: {str(e)}")
            return []
