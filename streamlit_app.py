"""
CORD-19 Data Explorer - Streamlit Application
Part 4: Interactive web application
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="CORD-19 Data Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('metadata.csv')
        return df
    except FileNotFoundError:
        st.error("metadata.csv file not found. Please ensure the file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def clean_data(df):
    """Clean and prepare the data"""
    if df is None:
        return None
    
    # Create a copy for cleaning
    cleaned_df = df.copy()
    
    # Remove rows where title is missing
    cleaned_df = cleaned_df.dropna(subset=['title'])
    
    # Convert date columns to datetime
    if 'publish_time' in cleaned_df.columns:
        cleaned_df['publish_time'] = pd.to_datetime(cleaned_df['publish_time'], errors='coerce')
        cleaned_df['year'] = cleaned_df['publish_time'].dt.year
        
        # Filter realistic years
        current_year = datetime.now().year
        cleaned_df = cleaned_df[
            (cleaned_df['year'] >= 1900) & 
            (cleaned_df['year'] <= current_year)
        ]
    
    # Create word count columns
    if 'abstract' in cleaned_df.columns:
        cleaned_df['abstract_word_count'] = cleaned_df['abstract'].fillna('').apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
    
    if 'title' in cleaned_df.columns:
        cleaned_df['title_word_count'] = cleaned_df['title'].fillna('').apply(
            lambda x: len(str(x).split()) if pd.notna(x) else 0
        )
    
    return cleaned_df

def get_word_frequency(text_series, top_n=20):
    """Get word frequency from text series"""
    all_text = ' '.join(text_series.fillna('').astype(str))
    words = re.findall(r'\b[a-zA-Z]{3,}\b', all_text.lower())
    
    # Common stop words to filter out
    stop_words = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they', 'been', 'have', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'would', 'there', 'could', 'other', 'after', 'first', 'well', 'way', 'many', 'these', 'may', 'then', 'them', 'two', 'more', 'very', 'what', 'know', 'just', 'see', 'him', 'take', 'than', 'only', 'think', 'also', 'back', 'use', 'her', 'our', 'out', 'day', 'get', 'has', 'had', 'his', 'how', 'man', 'new', 'now', 'old', 'any', 'may', 'say', 'she', 'should', 'some', 'such', 'make', 'over', 'here', 'even', 'most', 'state', 'years', 'year', 'system', 'being', 'study', 'used', 'using', 'based', 'analysis', 'data', 'results', 'research', 'method', 'methods', 'approach', 'model', 'models', 'case', 'cases', 'patients', 'patient', 'clinical', 'treatment', 'disease', 'health', 'medical', 'care', 'hospital', 'virus', 'viral', 'infection', 'covid', 'coronavirus', 'pandemic', 'epidemic', 'outbreak', 'response', 'public', 'population', 'community', 'social', 'economic', 'policy', 'policies', 'government', 'national', 'international', 'global', 'world', 'countries', 'country', 'united', 'states', 'china', 'europe', 'american', 'journal', 'article', 'paper', 'review', 'systematic', 'meta', 'literature', 'published', 'publication', 'author', 'authors', 'university', 'department', 'institute', 'center', 'hospital', 'school', 'college', 'medicine', 'science', 'sciences', 'biology', 'chemistry', 'physics', 'technology', 'engineering', 'computer', 'information', 'database', 'online', 'internet', 'web', 'digital', 'electronic', 'software', 'application', 'applications', 'tool', 'tools', 'platform', 'platforms', 'network', 'networks', 'communication', 'communications', 'media', 'news', 'report', 'reports', 'survey', 'surveys', 'interview', 'interviews', 'questionnaire', 'questionnaires', 'sample', 'samples', 'group', 'groups', 'control', 'controls', 'test', 'tests', 'testing', 'trial', 'trials', 'experiment', 'experiments', 'experimental', 'laboratory', 'lab', 'labs', 'clinical', 'hospital', 'medical', 'health', 'healthcare', 'medicine', 'pharmaceutical', 'drug', 'drugs', 'therapy', 'therapies', 'therapeutic', 'diagnosis', 'diagnostic', 'screening', 'prevention', 'preventive', 'vaccine', 'vaccines', 'vaccination', 'immunization', 'immunity', 'immune', 'antibody', 'antibodies', 'antigen', 'antigens', 'protein', 'proteins', 'gene', 'genes', 'genetic', 'genome', 'genomic', 'dna', 'rna', 'cell', 'cells', 'cellular', 'molecular', 'biology', 'biochemistry', 'microbiology', 'virology', 'epidemiology', 'pathology', 'physiology', 'anatomy', 'histology', 'cytology', 'immunology', 'pharmacology', 'toxicology', 'oncology', 'cardiology', 'neurology', 'psychiatry', 'psychology', 'sociology', 'anthropology', 'economics', 'statistics', 'statistical', 'mathematics', 'mathematical', 'computational', 'algorithm', 'algorithms', 'machine', 'learning', 'artificial', 'intelligence', 'deep', 'neural', 'network', 'networks'}
    
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    word_freq = Counter(filtered_words)
    
    return word_freq.most_common(top_n)

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🔬 CORD-19 Data Explorer")
    st.markdown("""
    **Simple exploration of COVID-19 research papers**
    
    This application provides an interactive analysis of the CORD-19 dataset, which contains 
    metadata about COVID-19 research papers. Use the sidebar controls to filter and explore the data.
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.stop()
    
    # Clean data
    with st.spinner("Cleaning data..."):
        cleaned_df = clean_data(df)
    
    if cleaned_df is None:
        st.error("Failed to clean data")
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("📊 Data Filters")
    
    # Year range filter
    if 'year' in cleaned_df.columns:
        year_min = int(cleaned_df['year'].min())
        year_max = int(cleaned_df['year'].max())
        
        # Default to recent years
        default_min = max(2019, year_min)
        default_max = min(2023, year_max)
        
        year_range = st.sidebar.slider(
            "Select year range",
            min_value=year_min,
            max_value=year_max,
            value=(default_min, default_max),
            step=1
        )
        
        # Filter data by year
        filtered_df = cleaned_df[
            (cleaned_df['year'] >= year_range[0]) & 
            (cleaned_df['year'] <= year_range[1])
        ]
    else:
        filtered_df = cleaned_df
        year_range = None
    
    # Journal filter
    if 'journal' in filtered_df.columns:
        top_journals = filtered_df['journal'].value_counts().head(20).index.tolist()
        selected_journals = st.sidebar.multiselect(
            "Select journals (optional)",
            options=top_journals,
            default=[]
        )
        
        if selected_journals:
            filtered_df = filtered_df[filtered_df['journal'].isin(selected_journals)]
    
    # Display basic statistics
    st.header("📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Papers", f"{len(filtered_df):,}")
    
    with col2:
        if 'year' in filtered_df.columns:
            unique_years = filtered_df['year'].nunique()
            st.metric("Years Covered", unique_years)
        else:
            st.metric("Years Covered", "N/A")
    
    with col3:
        if 'journal' in filtered_df.columns:
            unique_journals = filtered_df['journal'].nunique()
            st.metric("Unique Journals", unique_journals)
        else:
            st.metric("Unique Journals", "N/A")
    
    with col4:
        if 'abstract_word_count' in filtered_df.columns:
            avg_words = int(filtered_df['abstract_word_count'].mean())
            st.metric("Avg Abstract Length", f"{avg_words} words")
        else:
            st.metric("Avg Abstract Length", "N/A")
    
    # Visualizations
    st.header("📊 Data Visualizations")
    
    # Publications over time
    if 'year' in filtered_df.columns and len(filtered_df) > 0:
        st.subheader("Publications by Year")
        
        year_counts = filtered_df['year'].value_counts().sort_index()
        
        fig = px.bar(
            x=year_counts.index,
            y=year_counts.values,
            labels={'x': 'Year', 'y': 'Number of Publications'},
            title="Number of Publications by Year"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top journals
    if 'journal' in filtered_df.columns and len(filtered_df) > 0:
        st.subheader("Top Publishing Journals")
        
        journal_counts = filtered_df['journal'].value_counts().head(10)
        
        fig = px.bar(
            x=journal_counts.values,
            y=journal_counts.index,
            orientation='h',
            labels={'x': 'Number of Papers', 'y': 'Journal'},
            title="Top 10 Publishing Journals"
        )
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Word frequency analysis
    if 'title' in filtered_df.columns and len(filtered_df) > 0:
        st.subheader("Most Frequent Words in Titles")
        
        word_freq = get_word_frequency(filtered_df['title'], top_n=20)
        
        if word_freq:
            words, counts = zip(*word_freq)
            
            fig = px.bar(
                x=list(counts),
                y=list(words),
                orientation='h',
                labels={'x': 'Frequency', 'y': 'Word'},
                title="Top 20 Words in Paper Titles"
            )
            fig.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    # Abstract word count distribution
    if 'abstract_word_count' in filtered_df.columns and len(filtered_df) > 0:
        st.subheader("Abstract Length Distribution")
        
        word_counts = filtered_df['abstract_word_count']
        word_counts = word_counts[word_counts > 0]
        
        if len(word_counts) > 0:
            fig = px.histogram(
                x=word_counts,
                nbins=50,
                labels={'x': 'Word Count', 'y': 'Frequency'},
                title="Distribution of Abstract Word Counts"
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Papers by source
    if 'source_x' in filtered_df.columns and len(filtered_df) > 0:
        st.subheader("Papers by Source")
        
        source_counts = filtered_df['source_x'].value_counts().head(8)
        
        fig = px.pie(
            values=source_counts.values,
            names=source_counts.index,
            title="Distribution of Papers by Source"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Sample data display
    st.header("📋 Sample Data")
    
    # Display options
    col1, col2 = st.columns(2)
    
    with col1:
        num_rows = st.selectbox("Number of rows to display", [5, 10, 20, 50], index=1)
    
    with col2:
        if st.button("🔄 Refresh Sample"):
            st.rerun()
    
    # Display sample
    if len(filtered_df) > 0:
        sample_df = filtered_df.sample(min(num_rows, len(filtered_df)))
        
        # Select relevant columns for display
        display_columns = ['title', 'journal', 'publish_time', 'authors']
        available_columns = [col for col in display_columns if col in sample_df.columns]
        
        if available_columns:
            st.dataframe(
                sample_df[available_columns],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.dataframe(sample_df.head(num_rows), use_container_width=True)
    else:
        st.info("No data matches the current filters.")
    
    # Data summary
    st.header("📋 Data Summary")
    
    if len(filtered_df) > 0:
        st.write("**Filtered Dataset Information:**")
        
        summary_data = {
            "Metric": [
                "Total Papers",
                "Date Range",
                "Most Common Journal",
                "Average Title Length",
                "Papers with Abstracts"
            ],
            "Value": [
                f"{len(filtered_df):,}",
                f"{filtered_df['year'].min():.0f} - {filtered_df['year'].max():.0f}" if 'year' in filtered_df.columns else "N/A",
                filtered_df['journal'].mode().iloc[0] if 'journal' in filtered_df.columns and len(filtered_df['journal'].mode()) > 0 else "N/A",
                f"{filtered_df['title_word_count'].mean():.1f} words" if 'title_word_count' in filtered_df.columns else "N/A",
                f"{(filtered_df['abstract'].notna()).sum():,} ({(filtered_df['abstract'].notna()).mean()*100:.1f}%)" if 'abstract' in filtered_df.columns else "N/A"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this application:**
    - Built with Streamlit for interactive data exploration
    - Data source: CORD-19 dataset (COVID-19 research papers)
    - Use the sidebar filters to explore different subsets of the data
    - All visualizations update automatically based on your selections
    """)

if __name__ == "__main__":
    main()
