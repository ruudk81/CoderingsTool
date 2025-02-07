"""SQLite database operations for cache management"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from contextlib import contextmanager

from cache_config import CacheConfig


logger = logging.getLogger(__name__)


class CacheDatabase:
    """Manages SQLite database for cache metadata tracking"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.db_path = config.db_path
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database tables if they don't exist"""
        with self._get_connection() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    cache_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP,
                    file_size INTEGER,
                    processing_time FLOAT,
                    parameters TEXT,
                    config_hash TEXT,
                    status TEXT DEFAULT 'valid',
                    UNIQUE(filename, step_name)
                );
                
                CREATE TABLE IF NOT EXISTS processing_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    step_name TEXT NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    success BOOLEAN,
                    error_message TEXT,
                    input_hash TEXT,
                    output_hash TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_cache_filename_step 
                ON cache_metadata(filename, step_name);
                
                CREATE INDEX IF NOT EXISTS idx_cache_status 
                ON cache_metadata(status);
                
                CREATE INDEX IF NOT EXISTS idx_history_filename 
                ON processing_history(filename);
                
                CREATE TABLE IF NOT EXISTS clustering_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    overall_quality REAL,
                    silhouette_score REAL,
                    noise_ratio REAL,
                    coverage REAL,
                    mean_cluster_size REAL,
                    size_variance REAL,
                    num_clusters INTEGER,
                    num_meta_clusters INTEGER,
                    embedding_type TEXT,
                    language TEXT,
                    min_quality_score REAL,
                    max_noise_ratio REAL,
                    parameters TEXT,
                    attempts INTEGER
                );
                
                CREATE INDEX IF NOT EXISTS idx_metrics_filename 
                ON clustering_metrics(filename);
            ''')
    
    def record_cache_entry(self, 
                          filename: str, 
                          step_name: str,
                          cache_path: str,
                          file_hash: str,
                          file_size: int,
                          processing_time: float = None,
                          parameters: Dict = None,
                          config_hash: str = None) -> int:
        """Record a new cache entry or update existing one"""
        with self._get_connection() as conn:
            params_json = json.dumps(parameters) if parameters else None
            
            # Insert or replace the cache entry
            cursor = conn.execute('''
                INSERT OR REPLACE INTO cache_metadata 
                (filename, step_name, cache_path, file_hash, file_size, 
                 processing_time, parameters, config_hash, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (filename, step_name, cache_path, file_hash, file_size, 
                  processing_time, params_json, config_hash))
            
            return cursor.lastrowid
    
    def get_cache_info(self, filename: str, step_name: str) -> Optional[Dict]:
        """Get cache metadata for a specific file and step"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM cache_metadata 
                WHERE filename = ? AND step_name = ? AND status = 'valid'
            ''', (filename, step_name))
            
            row = cursor.fetchone()
            if row:
                # Convert row to dictionary
                result = dict(row)
                # Parse JSON fields
                if result['parameters']:
                    result['parameters'] = json.loads(result['parameters'])
                # Update last accessed time
                conn.execute('''
                    UPDATE cache_metadata 
                    SET last_accessed = CURRENT_TIMESTAMP 
                    WHERE id = ?
                ''', (result['id'],))
                return result
            return None
    
    def is_cache_valid(self, 
                      filename: str, 
                      step_name: str,
                      max_age_days: Optional[int] = None,
                      config_hash: Optional[str] = None) -> bool:
        """Check if cache entry is valid based on age and configuration"""
        cache_info = self.get_cache_info(filename, step_name)
        
        if not cache_info:
            return False
        
        # Check if cache file exists
        cache_path = Path(cache_info['cache_path'])
        if not cache_path.exists():
            self.invalidate_cache(filename, step_name)
            return False
        
        # Check age
        if max_age_days is None:
            max_age_days = self.config.max_cache_age_days
        
        created_at = datetime.fromisoformat(cache_info['created_at'])
        age = datetime.now() - created_at
        
        if age > timedelta(days=max_age_days):
            return False
        
        # Check configuration hash if provided
        if config_hash and cache_info.get('config_hash') != config_hash:
            return False
        
        return True
    
    def invalidate_cache(self, 
                        filename: Optional[str] = None, 
                        step_name: Optional[str] = None):
        """Mark cache entries as invalid"""
        with self._get_connection() as conn:
            if filename and step_name:
                # Invalidate specific entry
                conn.execute('''
                    UPDATE cache_metadata 
                    SET status = 'invalid' 
                    WHERE filename = ? AND step_name = ?
                ''', (filename, step_name))
            elif filename:
                # Invalidate all steps for a file
                conn.execute('''
                    UPDATE cache_metadata 
                    SET status = 'invalid' 
                    WHERE filename = ?
                ''', (filename,))
            elif step_name:
                # Invalidate all files for a step
                conn.execute('''
                    UPDATE cache_metadata 
                    SET status = 'invalid' 
                    WHERE step_name = ?
                ''', (step_name,))
            else:
                # Invalidate everything
                conn.execute("UPDATE cache_metadata SET status = 'invalid'")
    
    def record_processing(self,
                         filename: str,
                         step_name: str,
                         started_at: datetime,
                         completed_at: datetime = None,
                         success: bool = True,
                         error_message: str = None,
                         input_hash: str = None,
                         output_hash: str = None):
        """Record processing history"""
        with self._get_connection() as conn:
            conn.execute('''
                INSERT INTO processing_history 
                (filename, step_name, started_at, completed_at, 
                 success, error_message, input_hash, output_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (filename, step_name, started_at, completed_at,
                  success, error_message, input_hash, output_hash))
    
    def get_processing_history(self, 
                              filename: str = None, 
                              step_name: str = None,
                              limit: int = 100) -> List[Dict]:
        """Get processing history"""
        with self._get_connection() as conn:
            query = "SELECT * FROM processing_history WHERE 1=1"
            params = []
            
            if filename:
                query += " AND filename = ?"
                params.append(filename)
            
            if step_name:
                query += " AND step_name = ?"
                params.append(step_name)
            
            query += " ORDER BY started_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_entries(self, days_to_keep: int = None):
        """Remove old cache entries and their files"""
        if days_to_keep is None:
            days_to_keep = self.config.max_cache_age_days
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        with self._get_connection() as conn:
            # Get files to delete
            cursor = conn.execute('''
                SELECT cache_path FROM cache_metadata 
                WHERE created_at < ? OR status = 'invalid'
            ''', (cutoff_date,))
            
            files_to_delete = [row['cache_path'] for row in cursor.fetchall()]
            
            # Delete database entries
            conn.execute('''
                DELETE FROM cache_metadata 
                WHERE created_at < ? OR status = 'invalid'
            ''', (cutoff_date,))
            
            # Delete old processing history
            conn.execute('''
                DELETE FROM processing_history 
                WHERE started_at < ?
            ''', (cutoff_date,))
        
        # Delete actual files
        deleted_count = 0
        for file_path in files_to_delete:
            try:
                Path(file_path).unlink(missing_ok=True)
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
        
        return deleted_count
    
    def get_cache_statistics(self) -> Dict:
        """Get cache usage statistics"""
        with self._get_connection() as conn:
            stats = {}
            
            # Total cache entries
            cursor = conn.execute("SELECT COUNT(*) as count FROM cache_metadata WHERE status = 'valid'")
            stats['total_entries'] = cursor.fetchone()['count']
            
            # Cache size
            cursor = conn.execute("SELECT SUM(file_size) as total_size FROM cache_metadata WHERE status = 'valid'")
            stats['total_size_bytes'] = cursor.fetchone()['total_size'] or 0
            
            # Entries by step
            cursor = conn.execute('''
                SELECT step_name, COUNT(*) as count, SUM(file_size) as size 
                FROM cache_metadata 
                WHERE status = 'valid' 
                GROUP BY step_name
            ''')
            stats['by_step'] = {row['step_name']: {
                'count': row['count'], 
                'size': row['size'] or 0
            } for row in cursor.fetchall()}
            
            # Recent processing
            cursor = conn.execute('''
                SELECT COUNT(*) as count, 
                       SUM(CASE WHEN success THEN 1 ELSE 0 END) as success_count
                FROM processing_history 
                WHERE started_at > datetime('now', '-7 days')
            ''')
            row = cursor.fetchone()
            stats['recent_processing'] = {
                'total': row['count'],
                'successful': row['success_count']
            }
            
            return stats
    
    def record_clustering_metrics(self, 
                                filename: str, 
                                metrics: Dict,
                                config: 'ClusteringConfig',
                                attempts: int = 1) -> int:
        """Record clustering quality metrics"""
        with self._get_connection() as conn:
            # Extract metrics
            overall_quality = metrics.get('overall_quality', 0)
            silhouette_score = metrics.get('silhouette_score', 0)
            noise_ratio = metrics.get('noise_ratio', 0)
            coverage = metrics.get('coverage', 0)
            mean_cluster_size = metrics.get('mean_cluster_size', 0)
            size_variance = metrics.get('size_variance', 0)
            num_clusters = metrics.get('num_clusters', 0)
            num_meta_clusters = metrics.get('num_meta_clusters', 0)
            
            # Get additional metrics from config
            embedding_type = config.embedding_type
            language = config.language
            min_quality_score = config.min_quality_score
            max_noise_ratio = config.max_noise_ratio
            
            # Extract parameters used (min_samples, min_cluster_size)
            parameters = json.dumps({
                'min_samples': config.min_samples if hasattr(config, 'min_samples') else None,
                'min_cluster_size': config.min_cluster_size if hasattr(config, 'min_cluster_size') else None
            })
            
            cursor = conn.execute('''
                INSERT INTO clustering_metrics 
                (filename, overall_quality, silhouette_score, noise_ratio, coverage,
                 mean_cluster_size, size_variance, num_clusters, num_meta_clusters,
                 embedding_type, language, min_quality_score, max_noise_ratio, 
                 parameters, attempts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (filename, overall_quality, silhouette_score, noise_ratio, coverage,
                  mean_cluster_size, size_variance, num_clusters, num_meta_clusters,
                  embedding_type, language, min_quality_score, max_noise_ratio, 
                  parameters, attempts))
            
            return cursor.lastrowid
    
    def get_clustering_metrics(self, filename: str, limit: int = 1) -> List[Dict]:
        """Get clustering metrics for a file"""
        with self._get_connection() as conn:
            cursor = conn.execute('''
                SELECT * FROM clustering_metrics 
                WHERE filename = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (filename, limit))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result['parameters']:
                    result['parameters'] = json.loads(result['parameters'])
                results.append(result)
            
            return results