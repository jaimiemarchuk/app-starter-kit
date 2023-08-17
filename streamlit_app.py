import streamlit as st
import plotly.figure_factory as ff
import numpy as np
import pandas as pd 
from data_sci_toolkit.common_tools import config_tools

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

st.set_page_config(layout="wide")

# This code is different for each deployed app.
CURRENT_THEME = "dark"
IS_DARK_THEME = True
EXPANDER_TEXT = """

    ```python
    # Returns a dataframe containing all features/values and their attributes
    from data_sci_toolkit.common_tools import config_tools

    config_tools.get_data_dictionary()


    ```
    """
st.title("StellarAlgo Data Sci Dictionary!")

st.header("ya'know, for data science...")

with st.expander("What the heck is the point of a Data Sci Dictionary?"):
     st.markdown(
        """
        <div style="background-color: #2c918c; padding: 10px; border-radius: 10px;">
        
        The table below contains the stupendous data dictionary of the Data Sci Department.
        It includes every feature utilized in machine learning and data science processes, 
        along with attributes associated with these features, such as a feature's general description 
        and whether it was engineered or not. The ability to filter through the dataframe to find specific or 
        similar features is also available in the sidebar to the left. 

    
        </div>
        """,
        unsafe_allow_html=True
    )
with st.expander("How can I use this dictionary in my coding?"):
    st.write(EXPANDER_TEXT)

""
""


def filter_dataframe(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    print(key)

    with st.sidebar:
        modify = st.checkbox("Add filters or search for attributes in columns", key=key)

    if not modify:
        return df

    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    
    with st.sidebar:
        modification_container = st.container()

    with modification_container:
        to_filter_columns = df.iloc[:, :3]
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
    
            else:
                user_text_input = right.text_input(
                    f"Values in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]


    with st.sidebar:
        other_container = st.container()

    with other_container:
        to_filter_columns = df.iloc[:, -3:]
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
    
            else:
                user_text_input = right.text_input(
                    f"Values in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]


    return df

def draw_all(key, plot=False):


    df = config_tools.get_data_dictionary()

    feature_column = df.pop('feature')

    df.insert(0,'Feature',feature_column)


    feature_raw_column = df.pop('featureraw')

    df.insert(1,'Feature Raw', feature_raw_column)
    

    description_column = df.pop('description')

    df.insert(2,'Feature Description', description_column)


    st.dataframe(filter_dataframe(df, key))
 

draw_all("main", plot=True)


