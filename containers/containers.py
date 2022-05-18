from dependency_injector import providers, containers
from lib.db_store import DBStore
import os


class SqlConnector(containers.DeclarativeContainer):
    sql_connector = providers.Singleton(DBStore, 'postgresql+psycopg2://mlremoteuser:LJyRmZEfBae@94.130.201.172:5432/checknft')
