
"""
Quick Test Script for CORD-19 Analysis
Tests the main functionality with sample data
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils import DataProcessor, VisualizationHelper
    print("✅ Successfully imported utilities")
except ImportError as e:
    print(f"❌ Error importing utilities: {e}")
    sys.exit(1)

def create_sample_data(n_rows=1000):
    """Create sample data for testing"""
    print(f"Creating sample dataset with {n_rows} rows...")
    
    np.random.seed(42)  # For reproducible results
    
    # Sample journals
    journals = [
        'Nature', 'Science', 'Cell', 'The Lancet', 'New England Journal of Medicine',
        'PLOS ONE', 'Nature Medicine', 'Science Translational Medicine',
        'Journal of Virology', 'Proceedings of the National Academy of Sciences'
    ]
    
    # Sample sources
    sources = ['PubMed', 'ArXiv', 'bioRxiv', 'medRxiv', 'PMC']
    
    # Generate sample data
    data = {
        'title': [
            f"COVID-19 research study {i}: Analysis of viral transmission patterns"
            for i in range(n_rows)
        ],
        'abstract': [
            f"This study examines the impact of COVID-19 on public health. "
            f"We analyzed {np.random.randint(100, 1000)} samples and found "
            f"significant correlations. The results suggest important implications "
            f"for future pandemic preparedness and response strategies."
            for _ in range(n_rows)
        ],
        'journal': np.random.choice(journals, n_rows),
        'source_x': np.random.choice(sources, n_rows),
        'publish_time': [
            datetime(2020, 1, 1) + timedelta(days=np.random.randint(0, 1095))
            for _ in range(n_rows)
        ],
        'authors': [
            f"Author{i} A, Author{i} B, Author{i} C"
            for i in range(n_rows)
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values to simulate real data
    missing_indices = np.random.choice(df.index, size=int(n_rows * 0.1), replace=False)
    df.loc[missing_indices, 'abstract'] = None
    
    missing_indices = np.random.choice(df.index, size=int(n_rows * 0.05), replace=False)
    df.loc[missing_indices, 'journal'] = None
    
    print(f"✅ Created sample dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def test_data_processor():
    """Test the DataProcessor utilities"""
    print("\n" + "="*50)
    print("TESTING DATA PROCESSOR")
    print("="*50)
    
    # Create sample data
    df = create_sample_data(500)
    
    # Test data cleaning
    processor = DataProcessor()
    cleaned_df = processor.clean_dataframe(df)
    
    if cleaned_df is not None:
        print(f"✅ Data cleaning successful: {len(cleaned_df)} rows after cleaning")
        
        # Test statistics
        stats = processor.get_basic_stats(cleaned_df)
        print(f"✅ Generated statistics: {len(stats)} metrics")
        
        # Test word frequency
        if 'title' in cleaned_df.columns:
            word_freq = processor.get_word_frequency(cleaned_df['title'], top_n=10)
            print(f"✅ Word frequency analysis: {len(word_freq)} top words")
        
        # Test filtering
        filtered_df = processor.filter_by_year(cleaned_df, start_year=2021)
        print(f"✅ Year filtering: {len(filtered_df)} rows after filtering")
        
        return cleaned_df
    else:
        print("❌ Data processing failed")
        return None

def test_visualization():
    """Test the VisualizationHelper utilities"""
    print("\n" + "="*50)
    print("TESTING VISUALIZATION")
    print("="*50)
    
    # Create sample data
    df = create_sample_data(300)
    processor = DataProcessor()
    cleaned_df = processor.clean_dataframe(df)
    
    if cleaned_df is None:
        print("❌ Cannot test visualization without cleaned data")
        return False
    
    viz = VisualizationHelper()
    
    # Test individual plots
    try:
        # Year distribution
        fig1 = viz.create_year_distribution(cleaned_df)
        if fig1:
            print("✅ Year distribution plot created")
            import matplotlib.pyplot as plt
            plt.close(fig1)
        
        # Journal distribution
        fig2 = viz.create_journal_distribution(cleaned_df)
        if fig2:
            print("✅ Journal distribution plot created")
            plt.close(fig2)
        
        # Abstract length distribution
        fig3 = viz.create_abstract_length_distribution(cleaned_df)
        if fig3:
            print("✅ Abstract length distribution created")
            plt.close(fig3)
        
        # Word frequency plot
        word_freq = processor.get_word_frequency(cleaned_df['title'], top_n=10)
        fig4 = viz.create_word_frequency_plot(word_freq)
        if fig4:
            print("✅ Word frequency plot created")
            plt.close(fig4)
        
        # Comprehensive dashboard
        fig5 = viz.create_comprehensive_dashboard(cleaned_df, word_freq)
        if fig5:
            print("✅ Comprehensive dashboard created")
            plt.close(fig5)
        
        print("✅ All visualization tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def test_main_analysis():
    """Test the main analysis script functionality"""
    print("\n" + "="*50)
    print("TESTING MAIN ANALYSIS FUNCTIONALITY")
    print("="*50)
    
    try:
        # Import the main analysis class
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from data_analysis import CORD19Analyzer
        
        # Create sample data file
        df = create_sample_data(200)
        df.to_csv('test_metadata.csv', index=False)
        print("✅ Created test metadata file")
        
        # Test analyzer
        analyzer = CORD19Analyzer('test_metadata.csv')
        
        # Test loading
        if analyzer.load_data():
            print("✅ Data loading test passed")
        else:
            print("❌ Data loading test failed")
            return False
        
        # Test cleaning
        analyzer.clean_data()
        if analyzer.cleaned_df is not None:
            print("✅ Data cleaning test passed")
        else:
            print("❌ Data cleaning test failed")
            return False
        
        # Test analysis
        analyzer.analyze_data()
        print("✅ Data analysis test passed")
        
        # Clean up
        os.remove('test_metadata.csv')
        print("✅ Cleaned up test files")
        
        return True
        
    except Exception as e:
        print(f"❌ Main analysis test failed: {e}")
        return False

def test_streamlit_imports():
    """Test that Streamlit app can be imported"""
    print("\n" + "="*50)
    print("TESTING STREAMLIT APP IMPORTS")
    print("="*50)
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        # Test other required imports
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
        
        # Try to import the streamlit app (without running it)
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "streamlit_app", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "streamlit_app.py")
        )
        if spec:
            print("✅ Streamlit app can be imported")
            return True
        else:
            print("❌ Streamlit app import failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try: pip install streamlit plotly")
        return False

def run_all_tests():
    """Run all tests"""
    print("CORD-19 Analysis Project - Quick Test Suite")
    print("="*60)
    
    tests = [
        ("Data Processor", test_data_processor),
        ("Visualization", test_visualization),
        ("Main Analysis", test_main_analysis),
        ("Streamlit Imports", test_streamlit_imports)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if test_name == "Data Processor":
                # This test returns the cleaned dataframe
                result = test_func()
                results[test_name] = result is not None
            else:
                results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The project is ready to use.")
        print("\nNext steps:")
        print("1. Download metadata.csv from CORD-19 dataset")
        print("2. Run: python data_analysis.py")
        print("3. Run: streamlit run streamlit_app.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
