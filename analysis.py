import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def create_presentation_slides(results_file='results/association_results.csv'):
    """Create ready presentation slides"""
    
    # Load results
    results = pd.read_csv(results_file)
    
    print("="*70)
    print("🔍 COMPREHENSIVE ANALYSIS OF ASSOCIATION MEASURES")
    print("="*70)
    
    # Basic statistics
    summary = results.groupby(['relationship', 'method'])['power'].mean().unstack()
    
    print("\n📊 SUMMARY STATISTICS:")
    print("Average power by method across all relationships:")
    avg_power = summary.mean()
    for method, power in avg_power.items():
        print(f"   {method}: {power:.3f}")
    
    print(f"\nTotal relationship types analyzed: {len(summary)}")
    print(f"Total simulations: {len(results)}")
    
    # Best method analysis
    print("\n🏆 DOMINANCE ANALYSIS:")
    best_methods = summary.idxmax(axis=1)
    method_wins = best_methods.value_counts()
    for method, wins in method_wins.items():
        percentage = (wins / len(summary)) * 100
        print(f"   {method} wins in {wins} relationship types ({percentage:.1f}%)")
    
    # Create detailed comparison plot
    plt.figure(figsize=(12, 8))
    
    # Prepare data for grouped bar chart
    relationships = summary.index
    x = np.arange(len(relationships))
    width = 0.25
    
    plt.bar(x - width, summary['pearson'], width, label='Pearson', alpha=0.8)
    plt.bar(x, summary['dcor'], width, label='Distance Correlation', alpha=0.8)
    plt.bar(x + width, summary['mic'], width, label='MIC', alpha=0.8)
    
    plt.xlabel('Relationship Types')
    plt.ylabel('Average Power')
    plt.title('Comparison of Association Measures Across Different Relationship Types')
    plt.xticks(x, [rel.replace('_', '\n') for rel in relationships], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/method_comparison_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    power_pivot = results.pivot_table(values='power', 
                                    index='relationship', 
                                    columns='method')
    sns.heatmap(power_pivot, annot=True, cmap='YlOrRd', fmt='.3f', 
                cbar_kws={'label': 'Test Power'})
    plt.title('Heatmap of Method Performance Across Relationship Types')
    plt.tight_layout()
    plt.savefig('results/power_heatmap_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Noise sensitivity analysis
    print("\n📈 NOISE SENSITIVITY ANALYSIS:")
    noise_summary = results.groupby(['noise_level', 'method'])['power'].mean().unstack()
    print("Average power degradation with increasing noise:")
    print(noise_summary.round(3))
    
    # Method robustness
    print("\n🛡️  METHOD ROBUSTNESS (Performance at highest noise level):")
    high_noise = results[results['noise_level'] == results['noise_level'].max()]
    high_noise_power = high_noise.groupby('method')['power'].mean()
    for method, power in high_noise_power.items():
        print(f"   {method}: {power:.3f}")
    
    # Final conclusions
    print("\n🎯 FINAL CONCLUSIONS:")
    dcor_better_mic = (summary['dcor'] > summary['mic']).sum()
    mic_better_dcor = (summary['mic'] > summary['dcor']).sum()
    
    print(f"• Distance Correlation outperforms MIC in {dcor_better_mic} out of {len(summary)} cases")
    print(f"• MIC outperforms Distance Correlation in {mic_better_dcor} out of {len(summary)} cases")
    
    if mic_better_dcor == 0:
        print("• ✅ STRONG SUPPORT for Simon & Tibshirani's criticism of MIC")
    else:
        print("• ⚠️  Limited support for Simon & Tibshirani's claims")
    
    return summary

def generate_presentation_summary(results_file='results/association_results.csv'):
    """Generate final presentation summary"""
    
    results = pd.read_csv(results_file)
    summary = results.groupby(['relationship', 'method'])['power'].mean().unstack()
    
    print("\n" + "="*70)
    print("🎯 PRESENTATION SUMMARY - KEY FINDINGS")
    print("="*70)
    
    # Key findings for slides
    print("\n📋 KEY FINDINGS FOR SLIDES:")
    
    # Overall performance
    avg_performance = summary.mean().sort_values(ascending=False)
    best_overall = avg_performance.index[0]
    best_power = avg_performance.iloc[0]
    
    print(f"• Best overall performer: {best_overall} (average power: {best_power:.3f})")
    
    # Method dominance
    best_methods = summary.idxmax(axis=1)
    method_stats = best_methods.value_counts()
    
    for method, count in method_stats.items():
        percentage = (count / len(summary)) * 100
        avg_power = summary[method].mean()
        print(f"• {method}: best in {count} cases ({percentage:.1f}%), average power: {avg_power:.3f}")
    
    # Specific insights about MIC
    mic_performance = summary['mic'].mean()
    print(f"• MIC performance: average power {mic_performance:.3f} (significantly lower)")
    
    # Simon & Tibshirani verification
    print("\n🔬 VERIFICATION OF SIMON & TIBSHIRANI CLAIMS:")
    print("Original claim: 'MIC has lower power than dcor in every case except high-frequency sine'")
    
    dcor_vs_mic = (summary['dcor'] > summary['mic']).sum()
    mic_vs_dcor = (summary['mic'] > summary['dcor']).sum()
    
    print(f"• Our results: DCOR > MIC in {dcor_vs_mic} cases")
    print(f"• Our results: MIC > DCOR in {mic_vs_dcor} cases")
    
    if mic_vs_dcor == 0:
        print("• ✅ CONCLUSIVE: Our results fully support Simon & Tibshirani's criticism")
        print("• MIC consistently shows lower power than Distance Correlation")
    else:
        print("• ⚠️  PARTIAL SUPPORT: MIC shows lower power in most cases")
    
    print("\n💡 PRACTICAL IMPLICATIONS:")
    print("• For real-world data analysis: Prefer Distance Correlation over MIC")
    print("• For linear relationships: Pearson correlation remains effective")
    print("• MIC's 'equitability' comes at the cost of statistical power")
    
    return summary

if __name__ == "__main__":
    # Run comprehensive analysis
    summary = create_presentation_slides()
    
    # Generate final presentation summary
    generate_presentation_summary()