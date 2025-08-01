"""
Concurrency debugging utilities to identify bottlenecks in async processing.
This module provides detailed tracking of concurrent operations to pinpoint
where and why bottlenecks occur.
"""

import time
import asyncio
import psutil
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class ConcurrencySnapshot:
    """Snapshot of concurrency state at a point in time"""
    timestamp: float
    event_type: str
    identifier: str
    action: str  # 'start' or 'end'
    level: str  # 'batch', 'sub_batch', 'api_call'
    active_batches: int = 0
    active_sub_batches: int = 0
    active_api_calls: int = 0
    memory_usage_mb: float = 0
    cpu_percent: float = 0
    open_files: int = 0
    details: Dict[str, Any] = field(default_factory=dict)


class ConcurrencyDebugger:
    """
    Track and analyze concurrent operations to identify bottlenecks.
    Provides detailed metrics on what's running, when, and resource usage.
    """
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.start_time = time.time()
        
        # Concurrent operation counters
        self.active_batches = 0
        self.active_sub_batches = 0
        self.active_api_calls = 0
        
        # Track maximums
        self.max_concurrent_batches = 0
        self.max_concurrent_sub_batches = 0
        self.max_concurrent_api_calls = 0
        
        # Detailed tracking
        self.active_operations: Dict[str, Dict[str, Any]] = {
            'batches': {},
            'sub_batches': {},
            'api_calls': {}
        }
        
        # Timeline of all events
        self.timeline: List[ConcurrencySnapshot] = []
        
        # Performance metrics
        self.api_call_durations: List[float] = []
        self.batch_durations: List[float] = []
        self.sub_batch_durations: List[float] = []
        
        # Resource tracking
        self.process = psutil.Process(os.getpid())
        self.peak_memory_mb = 0
        self.peak_cpu_percent = 0
        
        # Bottleneck detection
        self.potential_bottlenecks: List[Dict[str, Any]] = []
        
    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current system resource usage"""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent(interval=0.01)
            open_files = len(self.process.open_files())
            
            # Track peaks
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
            self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
            
            return {
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent,
                'open_files': open_files
            }
        except:
            return {'memory_mb': 0, 'cpu_percent': 0, 'open_files': 0}
    
    def _create_snapshot(self, event_type: str, identifier: str, action: str, 
                        level: str, details: Optional[Dict] = None) -> ConcurrencySnapshot:
        """Create a snapshot of current concurrency state"""
        resources = self._get_resource_usage()
        
        snapshot = ConcurrencySnapshot(
            timestamp=time.time() - self.start_time,
            event_type=event_type,
            identifier=identifier,
            action=action,
            level=level,
            active_batches=self.active_batches,
            active_sub_batches=self.active_sub_batches,
            active_api_calls=self.active_api_calls,
            memory_usage_mb=resources['memory_mb'],
            cpu_percent=resources['cpu_percent'],
            open_files=resources['open_files'],
            details=details or {}
        )
        
        self.timeline.append(snapshot)
        return snapshot
    
    def _check_for_bottleneck(self, snapshot: ConcurrencySnapshot):
        """Check if current state indicates a bottleneck"""
        # High concurrency detection
        if self.active_batches > 50:
            self.potential_bottlenecks.append({
                'type': 'high_batch_concurrency',
                'timestamp': snapshot.timestamp,
                'active_batches': self.active_batches,
                'details': f"Very high batch concurrency: {self.active_batches} batches"
            })
        
        if self.active_api_calls > 100:
            self.potential_bottlenecks.append({
                'type': 'high_api_concurrency',
                'timestamp': snapshot.timestamp,
                'active_api_calls': self.active_api_calls,
                'details': f"Very high API concurrency: {self.active_api_calls} calls"
            })
        
        # Memory pressure detection
        if snapshot.memory_usage_mb > 1000:  # Over 1GB
            self.potential_bottlenecks.append({
                'type': 'high_memory_usage',
                'timestamp': snapshot.timestamp,
                'memory_mb': snapshot.memory_usage_mb,
                'details': f"High memory usage: {snapshot.memory_usage_mb:.1f} MB"
            })
    
    async def track_batch(self, batch_id: str, batch_size: int):
        """Context manager to track batch execution"""
        class BatchTracker:
            def __init__(self, debugger, batch_id, batch_size):
                self.debugger = debugger
                self.batch_id = batch_id
                self.batch_size = batch_size
                self.start_time = None
                
            async def __aenter__(self):
                self.start_time = time.time()
                self.debugger.active_batches += 1
                self.debugger.max_concurrent_batches = max(
                    self.debugger.max_concurrent_batches, 
                    self.debugger.active_batches
                )
                
                self.debugger.active_operations['batches'][self.batch_id] = {
                    'start_time': self.start_time,
                    'size': self.batch_size
                }
                
                snapshot = self.debugger._create_snapshot(
                    'batch', self.batch_id, 'start', 'batch',
                    {'batch_size': self.batch_size}
                )
                self.debugger._check_for_bottleneck(snapshot)
                
                return self
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.debugger.batch_durations.append(duration)
                self.debugger.active_batches -= 1
                
                if self.batch_id in self.debugger.active_operations['batches']:
                    del self.debugger.active_operations['batches'][self.batch_id]
                
                self.debugger._create_snapshot(
                    'batch', self.batch_id, 'end', 'batch',
                    {'duration': duration, 'batch_size': self.batch_size}
                )
        
        return BatchTracker(self, batch_id, batch_size)
    
    async def track_sub_batch(self, sub_batch_id: str, sub_batch_size: int):
        """Context manager to track sub-batch execution"""
        class SubBatchTracker:
            def __init__(self, debugger, sub_batch_id, sub_batch_size):
                self.debugger = debugger
                self.sub_batch_id = sub_batch_id
                self.sub_batch_size = sub_batch_size
                self.start_time = None
                
            async def __aenter__(self):
                self.start_time = time.time()
                self.debugger.active_sub_batches += 1
                self.debugger.max_concurrent_sub_batches = max(
                    self.debugger.max_concurrent_sub_batches,
                    self.debugger.active_sub_batches
                )
                
                self.debugger.active_operations['sub_batches'][self.sub_batch_id] = {
                    'start_time': self.start_time,
                    'size': self.sub_batch_size
                }
                
                snapshot = self.debugger._create_snapshot(
                    'sub_batch', self.sub_batch_id, 'start', 'sub_batch',
                    {'sub_batch_size': self.sub_batch_size}
                )
                self.debugger._check_for_bottleneck(snapshot)
                
                return self
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.debugger.sub_batch_durations.append(duration)
                self.debugger.active_sub_batches -= 1
                
                if self.sub_batch_id in self.debugger.active_operations['sub_batches']:
                    del self.debugger.active_operations['sub_batches'][self.sub_batch_id]
                
                self.debugger._create_snapshot(
                    'sub_batch', self.sub_batch_id, 'end', 'sub_batch',
                    {'duration': duration, 'sub_batch_size': self.sub_batch_size}
                )
        
        return SubBatchTracker(self, sub_batch_id, sub_batch_size)
    
    async def track_api_call(self, call_id: str, call_type: str = 'unknown'):
        """Context manager to track API call execution"""
        class APICallTracker:
            def __init__(self, debugger, call_id, call_type):
                self.debugger = debugger
                self.call_id = call_id
                self.call_type = call_type
                self.start_time = None
                
            async def __aenter__(self):
                self.start_time = time.time()
                self.debugger.active_api_calls += 1
                self.debugger.max_concurrent_api_calls = max(
                    self.debugger.max_concurrent_api_calls,
                    self.debugger.active_api_calls
                )
                
                self.debugger.active_operations['api_calls'][self.call_id] = {
                    'start_time': self.start_time,
                    'type': self.call_type
                }
                
                snapshot = self.debugger._create_snapshot(
                    'api_call', self.call_id, 'start', 'api_call',
                    {'call_type': self.call_type}
                )
                self.debugger._check_for_bottleneck(snapshot)
                
                return self
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.debugger.api_call_durations.append(duration)
                self.debugger.active_api_calls -= 1
                
                if self.call_id in self.debugger.active_operations['api_calls']:
                    del self.debugger.active_operations['api_calls'][self.call_id]
                
                self.debugger._create_snapshot(
                    'api_call', self.call_id, 'end', 'api_call',
                    {'duration': duration, 'call_type': self.call_type}
                )
        
        return APICallTracker(self, call_id, call_type)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive debugging report"""
        total_duration = time.time() - self.start_time
        
        # Calculate statistics
        avg_batch_duration = sum(self.batch_durations) / len(self.batch_durations) if self.batch_durations else 0
        avg_sub_batch_duration = sum(self.sub_batch_durations) / len(self.sub_batch_durations) if self.sub_batch_durations else 0
        avg_api_duration = sum(self.api_call_durations) / len(self.api_call_durations) if self.api_call_durations else 0
        
        report = {
            'module': self.module_name,
            'total_duration_seconds': total_duration,
            'concurrency_metrics': {
                'max_concurrent_batches': self.max_concurrent_batches,
                'max_concurrent_sub_batches': self.max_concurrent_sub_batches,
                'max_concurrent_api_calls': self.max_concurrent_api_calls,
                'total_batches': len(self.batch_durations),
                'total_sub_batches': len(self.sub_batch_durations),
                'total_api_calls': len(self.api_call_durations)
            },
            'performance_metrics': {
                'avg_batch_duration': avg_batch_duration,
                'avg_sub_batch_duration': avg_sub_batch_duration,
                'avg_api_call_duration': avg_api_duration,
                'longest_batch_duration': max(self.batch_durations) if self.batch_durations else 0,
                'longest_api_call': max(self.api_call_durations) if self.api_call_durations else 0
            },
            'resource_usage': {
                'peak_memory_mb': self.peak_memory_mb,
                'peak_cpu_percent': self.peak_cpu_percent
            },
            'potential_bottlenecks': self.potential_bottlenecks,
            'timeline_sample': self._get_timeline_sample()
        }
        
        return report
    
    def _get_timeline_sample(self) -> List[Dict[str, Any]]:
        """Get a sample of the timeline showing peak concurrency moments"""
        if not self.timeline:
            return []
        
        # Find moments of peak concurrency
        peak_moments = sorted(
            self.timeline,
            key=lambda s: s.active_batches + s.active_sub_batches + s.active_api_calls,
            reverse=True
        )[:10]
        
        return [
            {
                'timestamp': s.timestamp,
                'event': f"{s.event_type}:{s.action}",
                'active_operations': {
                    'batches': s.active_batches,
                    'sub_batches': s.active_sub_batches,
                    'api_calls': s.active_api_calls
                },
                'memory_mb': round(s.memory_usage_mb, 1),
                'cpu_percent': round(s.cpu_percent, 1)
            }
            for s in peak_moments
        ]
    
    def save_detailed_timeline(self, filepath: str):
        """Save the complete timeline for detailed analysis"""
        timeline_data = []
        for snapshot in self.timeline:
            timeline_data.append({
                'timestamp': snapshot.timestamp,
                'event_type': snapshot.event_type,
                'identifier': snapshot.identifier,
                'action': snapshot.action,
                'level': snapshot.level,
                'active_batches': snapshot.active_batches,
                'active_sub_batches': snapshot.active_sub_batches,
                'active_api_calls': snapshot.active_api_calls,
                'memory_usage_mb': snapshot.memory_usage_mb,
                'cpu_percent': snapshot.cpu_percent,
                'open_files': snapshot.open_files,
                'details': snapshot.details
            })
        
        with open(filepath, 'w') as f:
            json.dump({
                'module': self.module_name,
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'timeline': timeline_data
            }, f, indent=2)
        
        print(f"Detailed timeline saved to: {filepath}")