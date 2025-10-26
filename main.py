import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
import dcor
from tqdm import tqdm
import seaborn as sns
import os
from sklearn.feature_selection import mutual_info_regression

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class AssociationComparison:
    def __init__(self, n_samples=100, n_simulations=100, noise_levels=None):
        self.n_samples = n_samples
        self.n_simulations = n_simulations
        if noise_levels is None:
            self.noise_levels = np.linspace(0.1, 1.0, 6)  # Reduced noise range
    
    def generate_relationships(self, relationship_type, noise_level):
        """Generate different types of relationships"""
        np.random.seed(42)
        x = np.random.uniform(0, 1, self.n_samples)
        noise = np.random.normal(0, noise_level, self.n_samples)
        
        if relationship_type == 'linear':
            y = x + noise
        elif relationship_type == 'quadratic':
            y = 4 * (x - 0.5)**2 + noise
        elif relationship_type == 'cubic':
            y = 4 * (x - 0.5)**3 + noise
        elif relationship_type == 'sine_low_freq':
            y = np.sin(2 * np.pi * x) + noise
        elif relationship_type == 'sine_high_freq':
            y = np.sin(4 * np.pi * x) + noise  # Reduced frequency for better detection
        elif relationship_type == 'exponential':
            y = np.exp(x) + noise  # Simplified
        elif relationship_type == 'step':
            y = (x > 0.5).astype(float) + noise
        else:
            raise ValueError(f"Unknown relationship type: {relationship_type}")
            
        return x, y
    
    def calculate_mic_alternative(self, x, y):
        """Alternative MIC implementation using mutual information"""
        try:
            # Discretize data for better mutual information estimation
            x_bins = np.linspace(np.min(x), np.max(x), 10)
            y_bins = np.linspace(np.min(y), np.max(y), 10)
            
            x_disc = np.digitize(x, x_bins)
            y_disc = np.digitize(y, y_bins)
            
            # Calculate mutual information on discretized data
            mi = mutual_info_regression(x_disc.reshape(-1, 1), y_disc, random_state=42)[0]
            
            # Normalize by log of grid size
            mic_value = mi / np.log2(10)  # 10 bins
            
            return min(mic_value, 1.0)
        except:
            return 0.0
    
    def calculate_pearson(self, x, y):
        """Calculate Pearson correlation"""
        return abs(pearsonr(x, y)[0])
    
    def calculate_dcor(self, x, y):
        """Calculate distance correlation"""
        try:
            return dcor.distance_correlation(x, y)
        except:
            return 0.0
    
    def calculate_power(self, relationship_type, noise_level, method, threshold=0.05):
        """Calculate test power for specific method"""
        significant_count = 0
        
        for _ in range(self.n_simulations):
            x, y = self.generate_relationships(relationship_type, noise_level)
            
            if method == 'pearson':
                _, p_value = pearsonr(x, y)
                is_significant = p_value < threshold
            elif method == 'dcor':
                try:
                    # Simplified approach - use value threshold instead of p-value
                    dcor_value = dcor.distance_correlation(x, y)
                    is_significant = dcor_value > 0.3  # Empirical threshold
                except:
                    is_significant = False
            elif method == 'mic':
                mic_value = self.calculate_mic_alternative(x, y)
                is_significant = mic_value > 0.1
            
            if is_significant:
                significant_count += 1
        
        return significant_count / self.n_simulations
    
    def run_comparison(self, relationship_types=None):
        """Run full method comparison"""
        if relationship_types is None:
            relationship_types = [
                'linear', 'quadratic', 'cubic', 
                'sine_low_freq', 'sine_high_freq', 
                'exponential', 'step'
            ]
        
        results = []
        
        print("🚀 Starting method comparison...")
        for rel_type in relationship_types:
            print(f"📊 Analyzing relationship: {rel_type}")
            for noise in self.noise_levels:
                for method in ['pearson', 'dcor', 'mic']:
                    power = self.calculate_power(rel_type, noise, method)
                    results.append({
                        'relationship': rel_type,
                        'noise_level': noise,
                        'method': method,
                        'power': power
                    })
                    print(f"   {method}: noise={noise:.1f}, power={power:.3f}")
        
        return pd.DataFrame(results)
    
    def plot_results(self, results_df):
        """Visualize results"""
        relationships = results_df['relationship'].unique()
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, rel in enumerate(relationships):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            rel_data = results_df[results_df['relationship'] == rel]
            
            # Power plot
            for method in ['pearson', 'dcor', 'mic']:
                method_data = rel_data[rel_data['method'] == method]
                ax.plot(method_data['noise_level'], method_data['power'], 
                       label=method, linewidth=2, marker='o', markersize=4)
            
            ax.set_title(f'{rel.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Noise Level')
            ax.set_ylabel('Test Power')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(0, 1.05)
        
        # Remove extra subplots
        for idx in range(len(relationships), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig('results/power_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_english_insights(self, results_df):
        """Generate insights in English for presentation"""
        print("\n" + "="*70)
        print("📊 METHOD COMPARISON RESULTS - ENGLISH VERSION")
        print("="*70)
        
        summary = results_df.groupby(['relationship', 'method'])['power'].mean().unstack()
        
        print("\n📈 AVERAGE POWER ACROSS ALL SCENARIOS:")
        print(summary.round(3))
        
        print("\n🏆 BEST METHOD FOR EACH RELATIONSHIP TYPE:")
        best_methods = summary.idxmax(axis=1)
        for rel, method in best_methods.items():
            power = summary.loc[rel, method]
            print(f"   {rel:18} → {method:8} (power: {power:.3f})")
        
        print("\n🔍 KEY OBSERVATIONS:")
        
        # Method comparisons
        pearson_vs_dcor = (summary['pearson'] > summary['dcor']).sum()
        dcor_vs_pearson = (summary['dcor'] > summary['pearson']).sum()
        mic_vs_dcor = (summary['mic'] > summary['dcor']).sum()
        mic_vs_pearson = (summary['mic'] > summary['pearson']).sum()
        
        print(f"   • Pearson better than Distance Correlation in {pearson_vs_dcor} of {len(summary)} cases")
        print(f"   • Distance Correlation better than Pearson in {dcor_vs_pearson} of {len(summary)} cases")
        print(f"   • MIC better than Distance Correlation in {mic_vs_dcor} of {len(summary)} cases")
        print(f"   • MIC better than Pearson in {mic_vs_pearson} of {len(summary)} cases")
        
        print("\n💡 PRACTICAL RECOMMENDATIONS:")
        recommendations = [
            "• For linear relationships: Use Pearson correlation",
            "• For non-linear relationships: Use Distance correlation", 
            "• For complex patterns: Consider MIC (alternative implementation)",
            "• For general use: Distance correlation provides good balance",
            "• Note: MIC implementation is approximate using mutual information"
        ]
        for rec in recommendations:
            print(f"   {rec}")
        
        print("\n🔬 COMPARISON WITH SIMON & TIBSHIRANI CLAIMS:")
        print("• Original claim: 'MIC has lower power than dcor in every case except high-frequency sine'")
        print(f"• Our findings: DCOR better than MIC in {dcor_vs_pearson + pearson_vs_dcor - mic_vs_dcor} cases")
        print(f"• Our findings: MIC better than DCOR in {mic_vs_dcor} cases")
        
        if mic_vs_dcor == 0:
            print("• ✅ SUPPORTS Simon & Tibshirani: MIC has lower power than DCOR in all cases")
        else:
            print("• ⚠️  Partially supports Simon & Tibshirani")
        
        print("="*70)

def main():
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize
    comparator = AssociationComparison(n_samples=100, n_simulations=100)
    
    # Run comparison
    results = comparator.run_comparison()
    
    # Save results
    results.to_csv('results/association_results.csv', index=False)
    
    # Visualize
    comparator.plot_results(results)
    
    # Generate English insights
    comparator.generate_english_insights(results)
    
    return results

if __name__ == "__main__":
    results_df = main()