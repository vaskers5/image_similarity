from dependency_injector import providers, containers
from lib.db_store import DBStore
import os


class SqlConnector(containers.DeclarativeContainer):
    sql_connector = providers.Singleton(DBStore, os.getenv('DB_URI'))
