"""
Performance Monitoring and Adaptive Optimization for CoderingsTool
"""

import streamlit as st
import time
import psutil
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import json


@dataclass
class PerformanceMetric:
    operation: str
    duration: float
    items_processed: int
    batch_size: int
    memory_before: float
    memory_after: float
    timestamp: float
    success: bool = True
    error_msg: Optional[str] = None


class AdaptiveBatcher:
    """Automatically optimizes batch sizes based on performance history"""
    
    def __init__(self):
        self.performance_history = defaultdict(list)
        self.min_batch_size = 5
        self.max_batch_size = 100
        
    def record_performance(self, operation: str, batch_size: int, items: int, duration: float, success: bool = True):
        """Record performance metrics for an operation"""
        throughput = items / duration if duration > 0 else 0
        
        self.performance_history[operation].append({
            'batch_size': batch_size,
            'throughput': throughput,
            'duration': duration,
            'items': items,
            'timestamp': time.time(),
            'success': success
        })
        
        # Keep only recent history (last 20 records)
        if len(self.performance_history[operation]) > 20:
            self.performance_history[operation] = self.performance_history[operation][-20:]
    
    def get_optimal_batch_size(self, operation: str, default: int = 10) -> int:
        """Get optimal batch size based on performance history"""
        history = self.performance_history.get(operation, [])
        
        if len(history) < 3:  # Need at least 3 data points
            return default
            
        # Get successful operations only
        successful = [h for h in history if h.get('success', True)]
        if not successful:
            return default
            
        # Find batch size with best throughput
        best = max(successful, key=lambda x: x.get('throughput', 0))
        optimal_size = best['batch_size']
        
        # Ensure within reasonable bounds
        return max(self.min_batch_size, min(self.max_batch_size, optimal_size))
    
    def get_performance_summary(self, operation: str) -> Dict[str, Any]:
        """Get performance summary for an operation"""
        history = self.performance_history.get(operation, [])
        if not history:
            return {}
            
        successful = [h for h in history if h.get('success', True)]
        if not successful:
            return {"error": "No successful operations recorded"}
            
        throughputs = [h['throughput'] for h in successful]
        durations = [h['duration'] for h in successful]
        
        return {
            "total_operations": len(history),
            "successful_operations": len(successful),
            "avg_throughput": sum(throughputs) / len(throughputs),
            "best_throughput": max(throughputs),
            "avg_duration": sum(durations) / len(durations),
            "optimal_batch_size": self.get_optimal_batch_size(operation),
            "success_rate": len(successful) / len(history) * 100
        }


class PerformanceMonitor:
    """Comprehensive performance monitoring for the pipeline"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.adaptive_batcher = AdaptiveBatcher()
        self.start_times = {}
        self.operation_counts = defaultdict(int)
        
    @st.cache_resource
    def get_monitor(_self):
        """Get cached performance monitor instance"""
        return PerformanceMonitor()
    
    def start_operation(self, operation: str) -> str:
        """Start timing an operation"""
        operation_id = f"{operation}_{time.time()}"
        self.start_times[operation_id] = {
            'start_time': time.time(),
            'memory_before': self._get_memory_usage(),
            'operation': operation
        }
        return operation_id
    
    def end_operation(self, operation_id: str, items_processed: int = 1, batch_size: int = 1, success: bool = True, error_msg: str = None):
        """End timing an operation and record metrics"""
        if operation_id not in self.start_times:
            return
            
        start_info = self.start_times.pop(operation_id)
        end_time = time.time()
        duration = end_time - start_info['start_time']
        memory_after = self._get_memory_usage()
        
        metric = PerformanceMetric(
            operation=start_info['operation'],
            duration=duration,
            items_processed=items_processed,
            batch_size=batch_size,
            memory_before=start_info['memory_before'],
            memory_after=memory_after,
            timestamp=end_time,
            success=success,
            error_msg=error_msg
        )
        
        self.metrics.append(metric)
        self.operation_counts[start_info['operation']] += 1
        
        # Record for adaptive batching
        self.adaptive_batcher.record_performance(
            start_info['operation'], batch_size, items_processed, duration, success
        )
        
        # Keep only recent metrics (last 100)
        if len(self.metrics) > 100:
            self.metrics = self.metrics[-100:]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            return psutil.Process().memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        try:
            process = psutil.Process()
            return {
                "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "modules_loaded": len(sys.modules),
                "threads_count": process.num_threads(),
                "open_files": len(process.open_files()) if hasattr(process, 'open_files') else 0
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_pipeline_performance(self) -> Dict[str, Any]:
        """Get performance metrics for pipeline operations"""
        if not self.metrics:
            return {"message": "No performance data available"}
        
        # Group by operation
        by_operation = defaultdict(list)
        for metric in self.metrics:
            by_operation[metric.operation].append(metric)
        
        summary = {}
        for operation, metrics in by_operation.items():
            successful = [m for m in metrics if m.success]
            if successful:
                durations = [m.duration for m in successful]
                throughputs = [m.items_processed / m.duration for m in successful if m.duration > 0]
                memory_usage = [m.memory_after - m.memory_before for m in successful]
                
                summary[operation] = {
                    "total_runs": len(metrics),
                    "successful_runs": len(successful),
                    "success_rate": len(successful) / len(metrics) * 100,
                    "avg_duration": sum(durations) / len(durations),
                    "best_duration": min(durations),
                    "avg_throughput": sum(throughputs) / len(throughputs) if throughputs else 0,
                    "avg_memory_delta": sum(memory_usage) / len(memory_usage),
                    "optimal_batch_size": self.adaptive_batcher.get_optimal_batch_size(operation)
                }
        
        return summary
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get Streamlit cache performance metrics"""
        cache_info = {}
        
        # Try to get cache statistics from Streamlit
        try:
            # This is an approximation - Streamlit doesn't expose detailed cache metrics
            session_items = len([k for k in dir(st.session_state) if not k.startswith('_')])
            cache_info = {
                "session_state_items": session_items,
                "estimated_cache_entries": session_items * 2,  # Rough estimate
                "cache_types": ["@st.cache_resource", "@st.cache_data"]
            }
        except Exception as e:
            cache_info = {"error": f"Could not retrieve cache metrics: {e}"}
        
        return cache_info
    
    def display_performance_sidebar(self):
        """Display performance metrics in Streamlit sidebar"""
        if not st.sidebar.checkbox("📊 Performance Monitor", value=False):
            return
        
        st.sidebar.subheader("🔍 System Metrics")
        system_metrics = self.get_system_metrics()
        
        if "error" not in system_metrics:
            col1, col2 = st.sidebar.columns(2)
            with col1:
                st.metric("Memory", f"{system_metrics['memory_usage_mb']:.1f} MB")
                st.metric("Modules", system_metrics['modules_loaded'])
            with col2:
                st.metric("CPU", f"{system_metrics['cpu_percent']:.1f}%")
                st.metric("Threads", system_metrics['threads_count'])
        
        # Pipeline Performance
        st.sidebar.subheader("⚡ Pipeline Performance")
        pipeline_perf = self.get_pipeline_performance()
        
        if pipeline_perf and "message" not in pipeline_perf:
            for operation, metrics in pipeline_perf.items():
                with st.sidebar.expander(f"{operation} ({metrics['total_runs']} runs)"):
                    st.write(f"Success Rate: {metrics['success_rate']:.1f}%")
                    st.write(f"Avg Duration: {metrics['avg_duration']:.2f}s")
                    st.write(f"Best Duration: {metrics['best_duration']:.2f}s")
                    if metrics['avg_throughput'] > 0:
                        st.write(f"Throughput: {metrics['avg_throughput']:.1f} items/s")
                    st.write(f"Optimal Batch: {metrics['optimal_batch_size']}")
        
        # Cache Performance
        st.sidebar.subheader("💾 Cache Status")
        cache_metrics = self.get_cache_metrics()
        st.sidebar.json(cache_metrics)
        
        # Reset button
        if st.sidebar.button("🗑️ Clear Performance Data"):
            self.metrics.clear()
            self.adaptive_batcher.performance_history.clear()
            self.operation_counts.clear()
            st.sidebar.success("Performance data cleared!")
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest performance optimizations based on collected data"""
        suggestions = []
        
        system_metrics = self.get_system_metrics()
        if "error" not in system_metrics:
            # Memory suggestions
            memory_mb = system_metrics.get('memory_usage_mb', 0)
            if memory_mb > 2000:  # > 2GB
                suggestions.append("🔴 High memory usage detected. Consider clearing caches or processing smaller batches.")
            elif memory_mb > 1000:  # > 1GB
                suggestions.append("🟡 Moderate memory usage. Monitor for memory leaks in long sessions.")
            
            # Module loading suggestions
            modules = system_metrics.get('modules_loaded', 0)
            if modules > 2000:
                suggestions.append("🔴 Many modules loaded. Consider lazy loading for unused features.")
        
        # Pipeline performance suggestions
        pipeline_perf = self.get_pipeline_performance()
        for operation, metrics in pipeline_perf.items():
            if metrics.get('success_rate', 100) < 90:
                suggestions.append(f"🔴 {operation} has low success rate ({metrics['success_rate']:.1f}%). Check error handling.")
            
            if metrics.get('avg_duration', 0) > 60:  # > 1 minute
                suggestions.append(f"🟡 {operation} is slow ({metrics['avg_duration']:.1f}s avg). Consider batching or caching.")
        
        if not suggestions:
            suggestions.append("✅ Performance looks good! No immediate optimizations needed.")
        
        return suggestions


# Global performance monitor instance
@st.cache_resource
def get_performance_monitor():
    """Get cached performance monitor instance"""
    return PerformanceMonitor()