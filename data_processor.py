"""
Data Processing Utilities for CORD-19 Analysis
Helper functions for data cleaning and processing
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Utility class for data processing operations"""
    
    @staticmethod
    def load_metadata(file_path='metadata.csv', sample_size=None):
        """
        Load metadata.csv file with optional sampling
        
        Args:
            file_path (str): Path to metadata.csv file
            sample_size (int): Optional sample size for testing
            
        Returns:
            pd.DataFrame: Loaded dataframe
        """
        try:
            if sample_size:
                # Load a sample for testing
                df = pd.read_csv(file_path, nrows=sample_size)
                print(f"Loaded sample of {len(df)} rows for testing")
            else:
                df = pd.read_csv(file_path)
                print(f"Loaded full dataset with {len(df)} rows")
            
            return df
            
        except FileNotFoundError:
            print(f"Error: {file_path} not found.")
            print("Please download metadata.csv from the CORD-19 dataset.")
            return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    @staticmethod
    def clean_dataframe(df):
        """
        Clean and prepare the dataframe
        
        Args:
            df (pd.DataFrame): Raw dataframe
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        if df is None:
            return None
        
        print("Cleaning dataframe...")
        cleaned_df = df.copy()
        
        # Remove rows with missing titles
        initial_count = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=['title'])
        removed_count = initial_count - len(cleaned_df)
        if removed_count > 0:
            print(f"Removed {removed_count} rows with missing titles")
        
        # Process publication dates
        if 'publish_time' in cleaned_df.columns:
            cleaned_df['publish_time'] = pd.to_datetime(cleaned_df['publish_time'], errors='coerce')
            cleaned_df['year'] = cleaned_df['publish_time'].dt.year
            cleaned_df['month'] = cleaned_df['publish_time'].dt.month
            
            # Filter realistic years
            current_year = datetime.now().year
            before_filter = len(cleaned_df)
            cleaned_df = cleaned_df[
                (cleaned_df['year'] >= 1900) & 
                (cleaned_df['year'] <= current_year)
            ]
            after_filter = len(cleaned_df)
            if before_filter != after_filter:
                print(f"Filtered out {before_filter - after_filter} rows with unrealistic years")
        
        # Create derived columns
        DataProcessor._add_derived_columns(cleaned_df)
        
        print(f"Cleaning completed. Final dataset: {len(cleaned_df)} rows")
        return cleaned_df
    
    @staticmethod
    def _add_derived_columns(df):
        """Add derived columns to the dataframe"""
        
        # Abstract word count
        if 'abstract' in df.columns:
            df['abstract_word_count'] = df['abstract'].fillna('').apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )
        
        # Title word count
        if 'title' in df.columns:
            df['title_word_count'] = df['title'].fillna('').apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )
        
        # Has abstract flag
        if 'abstract' in df.columns:
            df['has_abstract'] = df['abstract'].notna() & (df['abstract'].str.len() > 0)
        
        # Author count (if authors column exists)
        if 'authors' in df.columns:
            df['author_count'] = df['authors'].fillna('').apply(
                lambda x: len([a.strip() for a in str(x).split(';') if a.strip()]) if pd.notna(x) and str(x) != '' else 0
            )
    
    @staticmethod
    def get_basic_stats(df):
        """
        Get basic statistics about the dataset
        
        Args:
            df (pd.DataFrame): Cleaned dataframe
            
        Returns:
            dict: Dictionary with basic statistics
        """
        if df is None or len(df) == 0:
            return {}
        
        stats = {
            'total_papers': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Year statistics
        if 'year' in df.columns:
            stats['year_range'] = {
                'min': int(df['year'].min()) if df['year'].notna().any() else None,
                'max': int(df['year'].max()) if df['year'].notna().any() else None,
                'unique_years': int(df['year'].nunique()) if df['year'].notna().any() else 0
            }
        
        # Journal statistics
        if 'journal' in df.columns:
            stats['journal_stats'] = {
                'unique_journals': int(df['journal'].nunique()),
                'top_journal': df['journal'].mode().iloc[0] if len(df['journal'].mode()) > 0 else None
            }
        
        # Abstract statistics
        if 'abstract_word_count' in df.columns:
            abstract_stats = df['abstract_word_count'][df['abstract_word_count'] > 0]
            if len(abstract_stats) > 0:
                stats['abstract_stats'] = {
                    'mean_length': float(abstract_stats.mean()),
                    'median_length': float(abstract_stats.median()),
                    'papers_with_abstract': int((df['abstract_word_count'] > 0).sum())
                }
        
        return stats
    
    @staticmethod
    def get_word_frequency(text_series, top_n=20, min_length=3):
        """
        Get word frequency from a text series
        
        Args:
            text_series (pd.Series): Series containing text
            top_n (int): Number of top words to return
            min_length (int): Minimum word length
            
        Returns:
            list: List of (word, count) tuples
        """
        # Combine all text
        all_text = ' '.join(text_series.fillna('').astype(str))
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
        
        # Filter by length
        words = [word for word in words if len(word) >= min_length]
        
        # Remove common stop words
        stop_words = DataProcessor._get_stop_words()
        filtered_words = [word for word in words if word not in stop_words]
        
        # Count frequency
        word_freq = Counter(filtered_words)
        
        return word_freq.most_common(top_n)
    
    @staticmethod
    def _get_stop_words():
        """Get common stop words to filter out"""
        return {
            'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they', 'been', 
            'have', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 
            'would', 'there', 'could', 'other', 'after', 'first', 'well', 'way', 'many', 
            'these', 'may', 'then', 'them', 'two', 'more', 'very', 'what', 'know', 'just', 
            'see', 'him', 'take', 'than', 'only', 'think', 'also', 'back', 'use', 'her', 
            'our', 'out', 'day', 'get', 'has', 'had', 'his', 'how', 'man', 'new', 'now', 
            'old', 'any', 'say', 'she', 'should', 'some', 'such', 'make', 'over', 'here', 
            'even', 'most', 'state', 'years', 'year', 'system', 'being', 'study', 'used', 
            'using', 'based', 'analysis', 'data', 'results', 'research', 'method', 'methods', 
            'approach', 'model', 'models', 'case', 'cases', 'patients', 'patient', 'clinical', 
            'treatment', 'disease', 'health', 'medical', 'care', 'hospital', 'virus', 'viral', 
            'infection', 'covid', 'coronavirus', 'pandemic', 'epidemic', 'outbreak', 'response', 
            'public', 'population', 'community', 'social', 'economic', 'policy', 'policies', 
            'government', 'national', 'international', 'global', 'world', 'countries', 'country'
        }
    
    @staticmethod
    def filter_by_year(df, start_year=None, end_year=None):
        """
        Filter dataframe by year range
        
        Args:
            df (pd.DataFrame): Input dataframe
            start_year (int): Start year (inclusive)
            end_year (int): End year (inclusive)
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        if df is None or 'year' not in df.columns:
            return df
        
        filtered_df = df.copy()
        
        if start_year is not None:
            filtered_df = filtered_df[filtered_df['year'] >= start_year]
        
        if end_year is not None:
            filtered_df = filtered_df[filtered_df['year'] <= end_year]
        
        return filtered_df
    
    @staticmethod
    def filter_by_journal(df, journals):
        """
        Filter dataframe by journal names
        
        Args:
            df (pd.DataFrame): Input dataframe
            journals (list): List of journal names
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        if df is None or 'journal' not in df.columns or not journals:
            return df
        
        return df[df['journal'].isin(journals)]
    
    @staticmethod
    def get_top_items(df, column, top_n=10):
        """
        Get top N items from a column
        
        Args:
            df (pd.DataFrame): Input dataframe
            column (str): Column name
            top_n (int): Number of top items
            
        Returns:
            pd.Series: Top items with counts
        """
        if df is None or column not in df.columns:
            return pd.Series()
        
        return df[column].value_counts().head(top_n)
    
    @staticmethod
    def export_summary(df, filename='data_summary.txt'):
        """
        Export data summary to text file
        
        Args:
            df (pd.DataFrame): Input dataframe
            filename (str): Output filename
        """
        if df is None:
            print("No data to export")
            return
        
        stats = DataProcessor.get_basic_stats(df)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("CORD-19 Dataset Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Papers: {stats.get('total_papers', 0):,}\n")
            
            if 'year_range' in stats:
                year_range = stats['year_range']
                f.write(f"Year Range: {year_range.get('min', 'N/A')} - {year_range.get('max', 'N/A')}\n")
                f.write(f"Unique Years: {year_range.get('unique_years', 0)}\n")
            
            if 'journal_stats' in stats:
                journal_stats = stats['journal_stats']
                f.write(f"Unique Journals: {journal_stats.get('unique_journals', 0):,}\n")
                f.write(f"Top Journal: {journal_stats.get('top_journal', 'N/A')}\n")
            
            if 'abstract_stats' in stats:
                abstract_stats = stats['abstract_stats']
                f.write(f"Papers with Abstract: {abstract_stats.get('papers_with_abstract', 0):,}\n")
                f.write(f"Average Abstract Length: {abstract_stats.get('mean_length', 0):.1f} words\n")
            
            f.write("\nTop 10 Journals:\n")
            top_journals = DataProcessor.get_top_items(df, 'journal', 10)
            for journal, count in top_journals.items():
                f.write(f"  {journal}: {count:,}\n")
            
            if 'title' in df.columns:
                f.write("\nTop 20 Words in Titles:\n")
                word_freq = DataProcessor.get_word_frequency(df['title'], 20)
                for word, count in word_freq:
                    f.write(f"  {word}: {count:,}\n")
        
        print(f"Summary exported to {filename}")

def main():
    """Test the data processor utilities"""
    print("Testing Data Processor Utilities")
    print("=" * 40)
    
    # Load sample data
    processor = DataProcessor()
    df = processor.load_metadata(sample_size=1000)  # Load sample for testing
    
    if df is not None:
        # Clean data
        cleaned_df = processor.clean_dataframe(df)
        
        # Get statistics
        stats = processor.get_basic_stats(cleaned_df)
        print("\nBasic Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Export summary
        processor.export_summary(cleaned_df, 'test_summary.txt')
        
        print("\nUtilities test completed successfully!")
    else:
        print("Could not load data for testing")

if __name__ == "__main__":
    main()
