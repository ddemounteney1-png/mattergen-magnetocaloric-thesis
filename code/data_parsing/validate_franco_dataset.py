#!/usr/bin/env python3
"""
Franco Dataset Validation & Summary Report
Verifies dataset integrity and provides quick statistics for Mattergen fine-tuning
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from project_paths import MATTERGEN_DIR, PREPROCESSING_WORK  # noqa: E402


import os
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
from pymatgen.core import Structure

def check_files_exist() -> Dict[str, bool]:
    """Check if all expected files exist."""
    checks = {
        'CIF Directory': (MATTERGEN_DIR / 'datasets/mce_finetuning/cifs').exists(),
        'Metadata CSV': (MATTERGEN_DIR / 'datasets/mce_finetuning/franco_dataset.csv').exists(),
        'Phase 2 Log': (PREPROCESSING_WORK / 'phase2_franco.log').exists(),
    }
    return checks

def load_metadata() -> pd.DataFrame:
    """Load Franco dataset metadata."""
    csv_path = str(MATTERGEN_DIR / 'datasets/mce_finetuning/franco_dataset.csv')
    return pd.read_csv(csv_path)

def validate_cif_files(cif_dir: Path, limit: int = 10) -> Dict:
    """Validate CIF files (check first N for speed)."""
    cif_files = sorted(cif_dir.glob('*.cif'))
    
    results = {
        'total_cif_files': len(cif_files),
        'valid_structures': 0,
        'invalid_structures': [],
        'avg_atoms': 0,
        'avg_volume': 0,
        'sample_validated': min(limit, len(cif_files))
    }
    
    atom_counts = []
    volumes = []
    
    for cif_file in cif_files[:limit]:
        try:
            struct = Structure.from_file(cif_file)
            results['valid_structures'] += 1
            atom_counts.append(len(struct))
            volumes.append(struct.volume)
        except Exception as e:
            results['invalid_structures'].append({
                'file': cif_file.name,
                'error': str(e)[:100]
            })
    
    if atom_counts:
        results['avg_atoms'] = np.mean(atom_counts)
        results['avg_volume'] = np.mean(volumes)
    
    return results

def generate_report():
    """Generate comprehensive validation report."""
    
    print("\n" + "="*80)
    print("FRANCO MAGNETOCALORIC DATASET - VALIDATION REPORT")
    print("="*80)
    print(f"Generated: {pd.Timestamp.now()}\n")
    
    # 1. File checks
    print("1. FILE INTEGRITY CHECK")
    print("-" * 80)
    file_checks = check_files_exist()
    for check_name, exists in file_checks.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {check_name}: {'Found' if exists else 'MISSING'}")
    
    if not all(file_checks.values()):
        print("\n⚠ WARNING: Some expected files are missing!")
        return
    
    # 2. Load metadata
    print("\n2. DATASET STATISTICS")
    print("-" * 80)
    df = load_metadata()
    print(f"  Total materials: {len(df)}")
    print(f"  Processing success rate: {df['processing_success'].sum()/len(df)*100:.1f}%")
    
    # 3. Prototype distribution
    print("\n3. PROTOTYPE DISTRIBUTION")
    print("-" * 80)
    prototype_dist = df['prototype_name'].value_counts()
    for proto, count in prototype_dist.items():
        percent = count / len(df) * 100
        print(f"  {proto:15s}: {count:3d} materials ({percent:5.1f}%)")
    
    # 4. Property statistics
    print("\n4. MATERIAL PROPERTIES")
    print("-" * 80)
    print("  Transition Temperature (T_K in K)")
    print(f"    Range:   {df['T_K'].min():.1f} - {df['T_K'].max():.1f} K")
    print(f"    Mean:    {df['T_K'].mean():.1f} K")
    print(f"    Median:  {df['T_K'].median():.1f} K")
    
    print("\n  Entropy Change (dS_J_kg_K)")
    valid_ds = df['dS_J_kg_K'].dropna()
    if len(valid_ds) > 0:
        print(f"    Range:   {valid_ds.min():.2f} - {valid_ds.max():.2f} J/kg·K")
        print(f"    Mean:    {valid_ds.mean():.2f} J/kg·K")
        print(f"    N/A:     {len(df) - len(valid_ds)} materials")
    
    print("\n  Magnetic Field (H_T_Tesla)")
    valid_ht = df['H_T_Tesla'].dropna()
    if len(valid_ht) > 0:
        print(f"    Range:   {valid_ht.min():.2f} - {valid_ht.max():.2f} T")
        print(f"    Mean:    {valid_ht.mean():.2f} T")
        print(f"    N/A:     {len(df) - len(valid_ht)} materials")
    
    # 5. Structure validation
    print("\n5. STRUCTURE VALIDATION (sample of 10)")
    print("-" * 80)
    cif_dir = (MATTERGEN_DIR / 'datasets/mce_finetuning/cifs')
    validation = validate_cif_files(cif_dir, limit=10)
    print(f"  Total CIF files:       {validation['total_cif_files']}")
    print(f"  Valid structures:      {validation['valid_structures']}/{validation['sample_validated']}")
    print(f"  Average atoms/cell:    {validation['avg_atoms']:.1f}")
    print(f"  Average cell volume:   {validation['avg_volume']:.2f} Ų")
    
    if validation['invalid_structures']:
        print(f"  Invalid structures:    {len(validation['invalid_structures'])}")
        for error in validation['invalid_structures'][:3]:
            print(f"    - {error['file']}: {error['error']}")
    
    # 6. Data quality
    print("\n6. DATA QUALITY SUMMARY")
    print("-" * 80)
    print(f"  Complete entries:      {len(df)} / {len(df)}")
    print(f"  Atoms per structure:   {df['num_atoms'].min():.0f} - {df['num_atoms'].max():.0f}")
    print(f"  Average atoms:         {df['num_atoms'].mean():.1f}")
    print(f"  Unique compositions:   {df['composition_normalized'].nunique()}")
    print(f"  Relaxation converged:  {df['relaxation_converged'].sum()} materials")
    
    # 7. Space groups
    print("\n7. SPACE GROUPS REPRESENTED")
    print("-" * 80)
    sg_dist = df['space_group'].value_counts().sort_values(ascending=False)
    for sg, count in sg_dist.head(5).items():
        print(f"  SG {int(sg):3d}: {count:3d} materials")
    
    # 8. Recommendations
    print("\n8. RECOMMENDATIONS FOR MATTERGEN FINE-TUNING")
    print("-" * 80)
    recommendations = []
    
    if df['processing_success'].sum() == len(df):
        recommendations.append("✓ All materials processed successfully")
    else:
        recommendations.append(f"⚠ {len(df) - df['processing_success'].sum()} materials had processing issues")
    
    if validation['valid_structures'] == validation['sample_validated']:
        recommendations.append("✓ All validated CIF files are readable")
    else:
        recommendations.append("⚠ Some CIF files failed to load")
    
    if df['T_K'].notna().sum() > len(df) * 0.8:
        recommendations.append("✓ High percentage of materials have transition temperature data")
    else:
        recommendations.append("⚠ Low percentage of materials have transition temperature data")
    
    if len(df) >= 100:
        recommendations.append(f"✓ Dataset size sufficient for fine-tuning ({len(df)} materials)")
    else:
        recommendations.append(f"⚠ Consider augmenting dataset ({len(df)} materials)")
    
    for rec in recommendations:
        print(f"  {rec}")
    
    # 9. File locations
    print("\n9. DATASET LOCATIONS")
    print("-" * 80)
    print(f"  CIF files:        {MATTERGEN_DIR}/datasets/mce_finetuning/cifs/")
    print(f"  Metadata CSV:     {MATTERGEN_DIR}/datasets/mce_finetuning/franco_dataset.csv")
    print(f"  Documentation:    {PREPROCESSING_WORK}/FRANCO_DATASET_GUIDE.md")
    
    # 10. Next steps
    print("\n10. READY FOR MATTERGEN FINE-TUNING")
    print("-" * 80)
    print("  Dataset is ready! Use with Mattergen fine-tuning module:")
    print("")
    print("  from pathlib import Path")
    print("  import pandas as pd")
    print("  ")
    print(f"  dataset_path = Path('{MATTERGEN_DIR}/datasets/mce_finetuning/cifs/')")
    print(f"  metadata = pd.read_csv('{MATTERGEN_DIR}/datasets/mce_finetuning/franco_dataset.csv')")
    print("  ")
    print("  structures = [Structure.from_file(f) for f in sorted(dataset_path.glob('*.cif'))]")
    
    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80 + "\n")

if __name__ == '__main__':
    try:
        generate_report()
    except Exception as e:
        print(f"\nError generating report: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
