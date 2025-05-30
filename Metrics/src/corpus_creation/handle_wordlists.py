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


def top_x(n, col, df, term_col='Term'):
    """Return top n rows sorted by col,
    including all tied values at the cutoff.

    Parameters:
    - n: int, number of top rows to return
    - col: str, column name to sort by
    - df: pandas DataFrame

    Returns:
    - DataFrame with top n rows plus any tied values at the cutoff
    """
    # Sort the dataframe by the specified column
    sorted_df = df.sort_values(
        by=[col, term_col], ascending=False
    ).reset_index(drop=True)

    if len(sorted_df) <= n:
        return sorted_df

    # Get the value at the nth position
    cutoff_value = sorted_df.iloc[n-1][col]

    # Return all rows with values >= cutoff_value
    return sorted_df[sorted_df[col] >= cutoff_value]


def top_x_with_core(
    n, col, df, core_terms,
    term_col='Term'
):
    """Return top n rows sorted by col,
    including all tied values at the cutoff.

    Make sure that core terms are included in the result
    but do not count towards the n.
    """
    # Drop core term rows
    df_no_core = df[~df[term_col].isin(core_terms)]

    # Get the top n rows from the non-core terms
    top_n_df = top_x(n, col, df_no_core, term_col)

    # Get existing core terms from the dataframe
    core_terms_df = df[df[term_col].isin(core_terms)]

    # Add any missing core terms with col=0
    missing_terms = set(core_terms) - set(df[term_col])
    if missing_terms:
        missing_df = pd.DataFrame(columns=df.columns)

        for term in missing_terms:
            new_row = {col_name: 0 for col_name in df.columns}
            new_row[term_col] = term
            missing_df = pd.concat(
                [missing_df, pd.DataFrame([new_row])], ignore_index=True
            )

        core_terms_df = pd.concat(
            [core_terms_df, missing_df], ignore_index=True
        )

    result_df = pd.concat([top_n_df, core_terms_df], ignore_index=True)
    result_df.sort_values(
        by=[col, term_col], ascending=False, inplace=True)
    result_df.reset_index(inplace=True, drop=True)

    return result_df
