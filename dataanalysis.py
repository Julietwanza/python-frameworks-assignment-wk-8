"""
CORD-19 Dataset Analysis Script
Part 1-3: Data Loading, Cleaning, and Analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from datetime import datetime
import numpy as np

class CORD19Analyzer:
    def __init__(self, data_path='metadata.csv'):
        """Initialize the analyzer with data path"""
        self.data_path = data_path
        self.df = None
        self.cleaned_df = None
        
    def load_data(self):
        """Part 1: Load and examine the data"""
        print("Loading CORD-19 metadata...")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully!")
            print(f"Dataset shape: {self.df.shape}")
            return True
        except FileNotFoundError:
            print(f"Error: {self.data_path} not found. Please download the metadata.csv file.")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def basic_exploration(self):
        """Part 1: Basic data exploration"""
        if self.df is None:
            print("Please load data first!")
            return
            
        print("\n=== BASIC DATA EXPLORATION ===")
        
        # Check DataFrame dimensions
        print(f"Dataset dimensions: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        # Data types
        print("\nData types:")
        print(self.df.dtypes)
        
        # Missing values
        print("\nMissing values per column:")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0].sort_values(ascending=False))
        
        # Basic statistics for numerical columns
        print("\nBasic statistics:")
        print(self.df.describe())
        
        # Column names
        print(f"\nColumn names ({len(self.df.columns)} total):")
        for i, col in enumerate(self.df.columns):
            print(f"{i+1}. {col}")
    
    def clean_data(self):
        """Part 2: Data cleaning and preparation"""
        if self.df is None:
            print("Please load data first!")
            return
            
        print("\n=== DATA CLEANING ===")
        
        # Create a copy for cleaning
        self.cleaned_df = self.df.copy()
        
        # Handle missing data
        print("Handling missing data...")
        
        # Remove rows where both title and abstract are missing
        initial_rows = len(self.cleaned_df)
        self.cleaned_df = self.cleaned_df.dropna(subset=['title'], how='all')
        print(f"Removed {initial_rows - len(self.cleaned_df)} rows with missing titles")
        
        # Convert date columns to datetime
        if 'publish_time' in self.cleaned_df.columns:
            print("Converting publish_time to datetime...")
            self.cleaned_df['publish_time'] = pd.to_datetime(self.cleaned_df['publish_time'], errors='coerce')
            
            # Extract year for analysis
            self.cleaned_df['year'] = self.cleaned_df['publish_time'].dt.year
            
            # Filter out unrealistic years (before 1900 or after current year)
            current_year = datetime.now().year
            self.cleaned_df = self.cleaned_df[
                (self.cleaned_df['year'] >= 1900) & 
                (self.cleaned_df['year'] <= current_year)
            ]
        
        # Create abstract word count column
        if 'abstract' in self.cleaned_df.columns:
            print("Creating abstract word count...")
            self.cleaned_df['abstract_word_count'] = self.cleaned_df['abstract'].fillna('').apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )
        
        # Create title word count column
        if 'title' in self.cleaned_df.columns:
            print("Creating title word count...")
            self.cleaned_df['title_word_count'] = self.cleaned_df['title'].fillna('').apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )
        
        print(f"Cleaned dataset shape: {self.cleaned_df.shape}")
        print("Data cleaning completed!")
    
    def analyze_data(self):
        """Part 3: Data analysis"""
        if self.cleaned_df is None:
            print("Please clean data first!")
            return
            
        print("\n=== DATA ANALYSIS ===")
        
        # Count papers by publication year
        if 'year' in self.cleaned_df.columns:
            print("\nPapers by publication year:")
            year_counts = self.cleaned_df['year'].value_counts().sort_index()
            print(year_counts.tail(10))  # Show last 10 years
        
        # Top journals
        if 'journal' in self.cleaned_df.columns:
            print("\nTop 10 journals publishing COVID-19 research:")
            journal_counts = self.cleaned_df['journal'].value_counts().head(10)
            print(journal_counts)
        
        # Most frequent words in titles
        if 'title' in self.cleaned_df.columns:
            print("\nMost frequent words in titles:")
            all_titles = ' '.join(self.cleaned_df['title'].fillna('').astype(str))
            # Simple word frequency (remove common words)
            words = re.findall(r'\b[a-zA-Z]{3,}\b', all_titles.lower())
            stop_words = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they', 'been', 'have', 'were', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'would', 'there', 'could', 'other', 'after', 'first', 'well', 'way', 'many', 'these', 'may', 'then', 'them', 'two', 'more', 'very', 'what', 'know', 'just', 'see', 'him', 'take', 'than', 'only', 'think', 'also', 'back', 'use', 'her', 'our', 'out', 'day', 'get', 'has', 'had', 'his', 'how', 'man', 'new', 'now', 'old', 'any', 'may', 'say', 'she', 'should', 'some', 'such', 'make', 'over', 'here', 'even', 'most', 'me', 'state', 'years', 'year', 'system', 'being', 'far', 'full', 'however', 'home', 'during', 'number', 'course', 'end', 'every', 'must', 'where', 'much', 'before', 'right', 'too', 'means', 'come', 'group', 'through', 'back', 'good', 'much', 'public', 'read', 'both', 'long', 'those', 'since', 'provide', 'might', 'between', 'study', 'show', 'large', 'often', 'together', 'follow', 'around', 'place', 'want', 'however', 'without', 'again', 'different', 'part', 'used', 'work', 'life', 'become', 'here', 'old', 'great', 'high', 'small', 'another', 'help', 'change', 'move', 'live', 'turn', 'start', 'might', 'seem', 'ask', 'point', 'late', 'want', 'try', 'kind', 'hand', 'picture', 'again', 'member', 'take', 'leave', 'name', 'same', 'important', 'while', 'mean', 'keep', 'student', 'team', 'eye', 'never', 'last', 'let', 'thought', 'city', 'tree', 'cross', 'since', 'hard', 'start', 'might', 'story', 'saw', 'far', 'sea', 'draw', 'left', 'late', 'run', 'dont', 'while', 'press', 'close', 'night', 'real', 'life', 'few', 'stop', 'open', 'seem', 'together', 'next', 'white', 'children', 'begin', 'got', 'walk', 'example', 'ease', 'paper', 'often', 'always', 'music', 'those', 'both', 'mark', 'book', 'letter', 'until', 'mile', 'river', 'car', 'feet', 'care', 'second', 'enough', 'plain', 'girl', 'usual', 'young', 'ready', 'above', 'ever', 'red', 'list', 'though', 'feel', 'talk', 'bird', 'soon', 'body', 'dog', 'family', 'direct', 'leave', 'song', 'measure', 'state', 'product', 'black', 'short', 'numeral', 'class', 'wind', 'question', 'happen', 'complete', 'ship', 'area', 'half', 'rock', 'order', 'fire', 'south', 'problem', 'piece', 'told', 'knew', 'pass', 'farm', 'top', 'whole', 'king', 'size', 'heard', 'best', 'hour', 'better', 'during', 'hundred', 'am', 'remember', 'step', 'early', 'hold', 'west', 'ground', 'interest', 'reach', 'fast', 'five', 'sing', 'listen', 'six', 'table', 'travel', 'less', 'morning', 'ten', 'simple', 'several', 'vowel', 'toward', 'war', 'lay', 'against', 'pattern', 'slow', 'center', 'love', 'person', 'money', 'serve', 'appear', 'road', 'map', 'science', 'rule', 'govern', 'pull', 'cold', 'notice', 'voice', 'fall', 'power', 'town', 'fine', 'certain', 'fly', 'unit', 'lead', 'cry', 'dark', 'machine', 'note', 'wait', 'plan', 'figure', 'star', 'box', 'noun', 'field', 'rest', 'correct', 'able', 'pound', 'done', 'beauty', 'drive', 'stood', 'contain', 'front', 'teach', 'week', 'final', 'gave', 'green', 'oh', 'quick', 'develop', 'sleep', 'warm', 'free', 'minute', 'strong', 'special', 'mind', 'behind', 'clear', 'tail', 'produce', 'fact', 'street', 'inch', 'lot', 'nothing', 'course', 'stay', 'wheel', 'full', 'force', 'blue', 'object', 'decide', 'surface', 'deep', 'moon', 'island', 'foot', 'yet', 'busy', 'test', 'record', 'boat', 'common', 'gold', 'possible', 'plane', 'age', 'dry', 'wonder', 'laugh', 'thousands', 'ago', 'ran', 'check', 'game', 'shape', 'yes', 'hot', 'miss', 'brought', 'heat', 'snow', 'bed', 'bring', 'sit', 'perhaps', 'fill', 'east', 'weight', 'language', 'among'}
            filtered_words = [word for word in words if word not in stop_words]
            word_freq = Counter(filtered_words)
            print("Top 20 words:")
            for word, count in word_freq.most_common(20):
                print(f"  {word}: {count}")
        
        # Papers by source
        if 'source_x' in self.cleaned_df.columns:
            print("\nPapers by source:")
            source_counts = self.cleaned_df['source_x'].value_counts().head(10)
            print(source_counts)
    
    def create_visualizations(self):
        """Part 3: Create visualizations"""
        if self.cleaned_df is None:
            print("Please clean data first!")
            return
            
        print("\n=== CREATING VISUALIZATIONS ===")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('CORD-19 Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Publications over time
        if 'year' in self.cleaned_df.columns:
            year_counts = self.cleaned_df['year'].value_counts().sort_index()
            # Filter to reasonable years (2000 onwards)
            year_counts = year_counts[year_counts.index >= 2000]
            
            axes[0, 0].bar(year_counts.index, year_counts.values, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('Publications by Year', fontweight='bold')
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Number of Publications')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Top journals
        if 'journal' in self.cleaned_df.columns:
            journal_counts = self.cleaned_df['journal'].value_counts().head(10)
            axes[0, 1].barh(range(len(journal_counts)), journal_counts.values, color='lightcoral', alpha=0.7)
            axes[0, 1].set_yticks(range(len(journal_counts)))
            axes[0, 1].set_yticklabels([j[:30] + '...' if len(j) > 30 else j for j in journal_counts.index])
            axes[0, 1].set_title('Top 10 Publishing Journals', fontweight='bold')
            axes[0, 1].set_xlabel('Number of Papers')
        
        # 3. Abstract word count distribution
        if 'abstract_word_count' in self.cleaned_df.columns:
            word_counts = self.cleaned_df['abstract_word_count']
            word_counts = word_counts[word_counts > 0]  # Remove zero counts
            word_counts = word_counts[word_counts < word_counts.quantile(0.95)]  # Remove outliers
            
            axes[1, 0].hist(word_counts, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Distribution of Abstract Word Counts', fontweight='bold')
            axes[1, 0].set_xlabel('Word Count')
            axes[1, 0].set_ylabel('Frequency')
        
        # 4. Papers by source
        if 'source_x' in self.cleaned_df.columns:
            source_counts = self.cleaned_df['source_x'].value_counts().head(8)
            axes[1, 1].pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Papers by Source', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('cord19_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'cord19_analysis.png'")
    
    def generate_word_cloud(self):
        """Generate word cloud from titles"""
        try:
            from wordcloud import WordCloud
            
            if 'title' in self.cleaned_df.columns:
                print("Generating word cloud...")
                
                # Combine all titles
                text = ' '.join(self.cleaned_df['title'].fillna('').astype(str))
                
                # Create word cloud
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    max_words=100,
                    colormap='viridis'
                ).generate(text)
                
                # Display word cloud
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud of Paper Titles', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig('wordcloud.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                print("Word cloud saved as 'wordcloud.png'")
            
        except ImportError:
            print("WordCloud library not installed. Skipping word cloud generation.")
            print("Install with: pip install wordcloud")
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting CORD-19 Dataset Analysis...")
        print("=" * 50)
        
        # Part 1: Load and explore data
        if not self.load_data():
            return False
        
        self.basic_exploration()
        
        # Part 2: Clean data
        self.clean_data()
        
        # Part 3: Analyze and visualize
        self.analyze_data()
        self.create_visualizations()
        self.generate_word_cloud()
        
        print("\n" + "=" * 50)
        print("Analysis completed successfully!")
        print("Generated files:")
        print("- cord19_analysis.png (main visualizations)")
        print("- wordcloud.png (word cloud of titles)")
        
        return True

def main():
    """Main function to run the analysis"""
    # Initialize analyzer
    analyzer = CORD19Analyzer('metadata.csv')
    
    # Run full analysis
    success = analyzer.run_full_analysis()
    
    if success:
        print("\nAnalysis completed! You can now run the Streamlit app with:")
        print("streamlit run streamlit_app.py")
    else:
        print("\nAnalysis failed. Please check that metadata.csv is in the current directory.")

if __name__ == "__main__":
    main()
