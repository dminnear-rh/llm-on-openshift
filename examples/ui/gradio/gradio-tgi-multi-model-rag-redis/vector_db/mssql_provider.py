import logging
import os
import re

import pyodbc
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_sqlserver import SQLServer_VectorStore

from vector_db.db_provider import DBProvider

logger = logging.getLogger(__name__)


class MSSQLProvider(DBProvider):
    type = "MSSQL"

    def __init__(self) -> None:
        super().__init__()
        connection_string = self._get_required_env("MSSQL_CONNECTION_STRING")
        table = self._get_required_env("MSSQL_TABLE")
        embedding_length = int(os.getenv("EMBEDDING_LENGTH", "768"))

        self.db = SQLServer_VectorStore(
            connection_string=connection_string,
            embedding_function=self.get_embeddings(),
            table_name=table,
            embedding_length=embedding_length,
        )

        # Extract and log server location (host,port)
        match = re.search(
            r"Server=([^;,]+)(?:,(\d+))?", connection_string, re.IGNORECASE
        )
        host = match.group(1) if match else "unknown"
        port = match.group(2) if match and match.group(2) else "default"

        logger.info(
            "Connected to MSSQL vector store at %s:%s (table: %s)",
            host,
            port,
            table,
        )

        # Optional: log the number of vector records
        try:
            with pyodbc.connect(connection_string, timeout=5) as conn:
                cursor = conn.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
                row_count = cursor.fetchone()[0]
                logger.info("Vector table '%s' contains %d records", table, row_count)
        except Exception as e:
            logger.warning("Could not count records in table '%s': %s", table, e)

    @staticmethod
    def _get_required_env(key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"{key} environment variable is required.")
        return value

    @classmethod
    def _get_type(cls) -> str:
        """Returns type of the db provider."""
        return cls.type

    def get_retriever(self) -> VectorStoreRetriever:
        return self.db.as_retriever()
