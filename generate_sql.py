import os
import sqlite3
from contextlib import closing

from pathlib import Path

from sqlalchemy import MetaData, create_engine
from sqlacodegen.generators import DeclarativeGenerator


TMP_DB_NAME = "tmp.db"


if __name__ == "__main__":
    with closing(sqlite3.connect(TMP_DB_NAME)) as connection:
        with connection:
            connection.executescript(Path("schema.sql").read_text())

    engine = create_engine(f"sqlite:///{TMP_DB_NAME}")
    try:
        metadata = MetaData()
        metadata.reflect(bind=engine, views=True)

        generator = DeclarativeGenerator(metadata, engine, options=[])
        source = generator.generate()

        Path("sql_models.py").write_text(source, encoding="utf-8")
    finally:
        engine.dispose()

    os.remove(TMP_DB_NAME)
