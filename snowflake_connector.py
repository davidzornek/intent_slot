from typing import List, Optional, Union

import pandas as pd
import snowflake.connector as sf
from snowflake.connector.errors import NotSupportedError
from snowflake.connector.pandas_tools import write_pandas
from sqlalchemy import create_engine


class SnowflakeConnector:
    def __init__(
        self,
        username: str,
        password: str,
    ):
        self.account = ""
        self.warehouse = ""
        self.db_name = ""
        self.schema = ""
        self.username = username
        self.password = password
        self.connnection = None
        self.cursor = None

    def connect(self):
        """Connects to the snowflake db."""
        self.connection = sf.connect(
            user=self.username,
            password=self.password,
            account=self.account,
            warehouse=self.warehouse,
            database=self.db_name,
            schema=self.schema,
        )
        self.cursor = self.connection.cursor()

    def create_table_from_df(
        self,
        df: pd.DataFrame,
        table_name: str,
        description: Optional[str] = None,
        normalize_column_names: bool = True,
    ) -> dict:
        """Automatically creates a new snowflake table from a pandas dataframe.

        Params:
            df: a dataframe
            table_name: the name of the new table
            description: An optional description of the dataset that will be added
            as a comment on the table in Snowflake.
            normalize_column_names: whether to normalize column names to snake case.

        Returns:
            dict:
                Indicates success, number of chunks, and number of rows.
        """
        if normalize_column_names:
            df.columns = [x.lower().replace(" ", "_") for x in df.columns]

        # pandas will use sqlite flavor if we don't wrap snowflake in sqlalchemy,
        # causing unwanted dtype conversions
        sqlalchemy_engine = create_engine(
            "snowflake://{user}:{password}@{account_identifier}/".format(
                user=self.username,
                password=self.password,
                account_identifier=self.account,
            )
        )

        df_schema = pd.io.sql.get_schema(df, table_name, con=sqlalchemy_engine)
        df_schema = df_schema.replace(
            f'CREATE TABLE "{table_name}"',
            f'CREATE TABLE "{self.db_name}"."{self.schema}"."{table_name}"',
        )
        for col in df.columns:
            df_schema = df_schema.replace(f"{col} ", f'"{col}" ')
        self.sql_query(df_schema)

        if description is not None:
            comment_query = f"COMMENT ON TABLE {table_name} IS '{description}'"
            self.sql_query(comment_query)

        success, nchunks, nrows, _ = write_pandas(self.connection, df, table_name)
        return {"success": success, "nchunks": nchunks, "nrows": nrows}

    def get_table(self, table_name: str) -> pd.DataFrame:
        """Pulls an entire snowflake table into a pandas data frame.

        Params:
            table_name: name of the table to pull
        """
        sql_query = f"SELECT * FROM {table_name}"
        return self.sql_query(sql_query)

    def sql_query(self, sql_query: str) -> Union[List[tuple], pd.DataFrame]:
        """Executes a raw sql query and returns results as a pandas data frame.
        For sql operations that do not return tabular data, a list of tuples
        indicating success or failure is returned instead.

        Params:
            sql_query: a sql query

        Returns:
            Results of the query as a data frame or a list of tuples
            with messages about success.
        """
        self.cursor.execute(sql_query)
        try:
            return self.cursor.fetch_pandas_all()
        except NotSupportedError:
            return self.cursor.fetchall()

