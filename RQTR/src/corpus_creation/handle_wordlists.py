import pandas as pd


def remove_redundant(df: pd.DataFrame, check_col='Term') -> pd.DataFrame:
    """
    Remove redundant columns from a DataFrame.

    A redundant row is one that is already covered by another row
    above it in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to remove redundant columns from.

    Returns
    -------
    pd.DataFrame
        The DataFrame with redundant columns removed.
    """

    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    seen_ngrams = set()
    rows_to_remove = []
    for index, row in df_copy.iterrows():
        ngram = row[check_col]
        sub_ngrams = []
        for n in range(1, len(ngram) + 1):
            for i in range(len(ngram) - n + 1):
                sub_ngrams.append(ngram[i:i + n])
        for sub_ngram in sub_ngrams:
            if sub_ngram in seen_ngrams:
                rows_to_remove.append(index)
                break
        seen_ngrams.add(ngram)

    # Remove the rows that are marked for removal
    df_copy.drop(rows_to_remove, inplace=True)
    df_copy.reset_index(drop=True, inplace=True)

    return df_copy


def top_x(n, col, df):
    """
    Return top n rows sorted by col, including all tied values at the cutoff.

    Parameters:
    - n: int, number of top rows to return
    - col: str, column name to sort by
    - df: pandas DataFrame

    Returns:
    - DataFrame with top n rows plus any tied values at the cutoff
    """
    # Sort the dataframe by the specified column
    sorted_df = df.sort_values(by=col, ascending=False).reset_index(drop=True)

    if len(sorted_df) <= n:
        return sorted_df

    # Get the value at the nth position
    cutoff_value = sorted_df.iloc[n-1][col]

    # Return all rows with values >= cutoff_value
    return sorted_df[sorted_df[col] >= cutoff_value]
