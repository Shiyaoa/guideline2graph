"""
映射缓存数据库

使用SQLite存储已验证的术语映射，避免重复处理
"""

import sqlite3
from typing import Optional, Dict, List
from datetime import datetime


class MappingCache:
    """映射缓存数据库"""

    def __init__(self, db_path: str = "term_mapping_cache.db"):
        """
        初始化缓存

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # 主映射表
        c.execute('''
            CREATE TABLE IF NOT EXISTS term_mapping (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chinese_term TEXT NOT NULL,
                english_term TEXT,
                concept_id INTEGER NOT NULL,
                concept_name TEXT,
                domain_id TEXT,
                vocabulary_id TEXT,
                extraction_confidence REAL,
                match_score REAL,
                review_confidence REAL,
                final_confidence REAL,
                match_type TEXT,
                is_verified INTEGER DEFAULT 0,
                source TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                UNIQUE(chinese_term, domain_id)
            )
        ''')

        # 索引
        c.execute('CREATE INDEX IF NOT EXISTS idx_chinese_term ON term_mapping(chinese_term)')
        c.execute('CREATE INDEX IF NOT EXISTS idx_concept_id ON term_mapping(concept_id)')

        # 待审核队列
        c.execute('''
            CREATE TABLE IF NOT EXISTS review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chinese_term TEXT NOT NULL,
                english_term TEXT,
                suggested_concept_id INTEGER,
                suggested_concept_name TEXT,
                domain_id TEXT,
                confidence REAL,
                status TEXT DEFAULT 'pending',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                reviewed_at TEXT,
                reviewer_notes TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def get(self, chinese_term: str, domain_id: Optional[str] = None) -> Optional[Dict]:
        """
        查询缓存

        Args:
            chinese_term: 中文术语
            domain_id: 可选的领域限制

        Returns:
            缓存的映射记录，未找到返回None
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        if domain_id:
            c.execute('''
                SELECT chinese_term, english_term, concept_id, concept_name,
                       domain_id, vocabulary_id, extraction_confidence,
                       match_score, review_confidence, final_confidence,
                       match_type, is_verified, source
                FROM term_mapping
                WHERE chinese_term = ? AND domain_id = ?
            ''', (chinese_term, domain_id))
        else:
            c.execute('''
                SELECT chinese_term, english_term, concept_id, concept_name,
                       domain_id, vocabulary_id, extraction_confidence,
                       match_score, review_confidence, final_confidence,
                       match_type, is_verified, source
                FROM term_mapping
                WHERE chinese_term = ?
            ''', (chinese_term,))

        row = c.fetchone()
        conn.close()

        if row:
            return {
                'chinese_term': row[0],
                'english_term': row[1],
                'concept_id': row[2],
                'concept_name': row[3],
                'domain_id': row[4],
                'vocabulary_id': row[5],
                'extraction_confidence': row[6],
                'match_score': row[7],
                'review_confidence': row[8],
                'final_confidence': row[9],
                'match_type': row[10],
                'is_verified': bool(row[11]),
                'source': row[12],
            }
        return None

    def save(self, result: Dict, source: str = ""):
        """
        保存映射到缓存

        Args:
            result: 映射结果字典
            source: 来源标识
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            INSERT OR REPLACE INTO term_mapping
            (chinese_term, english_term, concept_id, concept_name, domain_id,
             vocabulary_id, extraction_confidence, match_score, review_confidence,
             final_confidence, match_type, source, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result['chinese_term'],
            result.get('english_term', ''),
            result.get('concept_id', 0),
            result.get('concept_name', ''),
            result.get('domain_id', ''),
            result.get('vocabulary_id', ''),
            result.get('extraction_confidence', 0),
            result.get('match_score', 0),
            result.get('review_confidence', 0),
            result.get('final_confidence', 0),
            result.get('match_type', ''),
            source,
            datetime.now().isoformat()
        ))

        conn.commit()
        conn.close()

    def add_to_review_queue(self, item: Dict):
        """添加到待审核队列"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            INSERT INTO review_queue
            (chinese_term, english_term, suggested_concept_id,
             suggested_concept_name, domain_id, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            item['chinese_term'],
            item.get('english_term', ''),
            item.get('concept_id', 0),
            item.get('concept_name', ''),
            item.get('domain_id', ''),
            item.get('confidence', 0)
        ))

        conn.commit()
        conn.close()

    def get_review_queue(self, status: str = 'pending') -> List[Dict]:
        """获取待审核队列"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        c.execute('''
            SELECT id, chinese_term, english_term, suggested_concept_id,
                   suggested_concept_name, domain_id, confidence, created_at
            FROM review_queue
            WHERE status = ?
            ORDER BY confidence DESC, created_at ASC
        ''', (status,))

        rows = c.fetchall()
        conn.close()

        return [{
            'id': row[0],
            'chinese_term': row[1],
            'english_term': row[2],
            'suggested_concept_id': row[3],
            'suggested_concept_name': row[4],
            'domain_id': row[5],
            'confidence': row[6],
            'created_at': row[7]
        } for row in rows]

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()

        # 总映射数
        c.execute('SELECT COUNT(*) FROM term_mapping')
        total_mappings = c.fetchone()[0]

        # 已验证数
        c.execute('SELECT COUNT(*) FROM term_mapping WHERE is_verified = 1')
        verified_mappings = c.fetchone()[0]

        # 按领域统计
        c.execute('SELECT domain_id, COUNT(*) FROM term_mapping GROUP BY domain_id')
        by_domain = dict(c.fetchall())

        # 按匹配类型统计
        c.execute('SELECT match_type, COUNT(*) FROM term_mapping GROUP BY match_type')
        by_match_type = dict(c.fetchall())

        # 待审核数
        c.execute('SELECT COUNT(*) FROM review_queue WHERE status = "pending"')
        pending_reviews = c.fetchone()[0]

        conn.close()

        return {
            'total_mappings': total_mappings,
            'verified_mappings': verified_mappings,
            'pending_reviews': pending_reviews,
            'by_domain': by_domain,
            'by_match_type': by_match_type,
        }

    def export_to_dataframe(self):
        """导出为DataFrame"""
        import pandas as pd
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM term_mapping', conn)
        conn.close()
        return df

    def clear_all(self):
        """清空所有缓存（谨慎使用）"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('DELETE FROM term_mapping')
        c.execute('DELETE FROM review_queue')
        conn.commit()
        conn.close()