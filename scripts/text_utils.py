import pandas as pd

def create_dynamic_embedding_text(df, selected_fields, options):
    """
    Create text for embedding by combining selected fields.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        selected_fields (list): List of field names to combine
        options (dict): Configuration options including:
            - join_option (bool): Whether to join fields with labels
            - separator (str): Separator to use between fields
    
    Returns:
        pandas.Series: Combined text series
    """
    texts = []
    for _, row in df.iterrows():
        if options['join_option']:
            # Join with field names as labels
            field_texts = [f"{field}: {str(row[field])}" for field in selected_fields]
        else:
            # Join values only
            field_texts = [str(row[field]) for field in selected_fields]
        
        texts.append(options['separator'].join(field_texts))
    
    return pd.Series(texts)