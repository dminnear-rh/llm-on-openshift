import os

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_sqlserver import SQLServer_VectorStore
from vector_db.db_provider import DBProvider


class MSSQLProvider(DBProvider):
    type = "MSSQL"

    def __init__(self) -> None:
        super().__init__()
        connection_string = self._get_required_env("MSSQL_CONNECTION_STRING")
        table = self._get_required_env("MSSQL_TABLE")
        self.db = SQLServer_VectorStore(
            connection_string=connection_string,
            embedding_function=self.get_embeddings(),
            table_name=table,
            embedding_length=int(os.getenv("EMBEDDING_LENGTH", "768")),
        )

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
