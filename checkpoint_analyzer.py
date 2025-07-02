#!/usr/bin/env python
"""
Checkpoint Architecture Analyzer for RETFound Models

This script analyzes and compares two RETFound checkpoints to identify 
architectural differences that might cause saliency map generation issues.

Usage:
python checkpoint_analyzer.py \
    --checkpoint1 /NVME/scratch/dave/VD_fold_1/fold1/checkpoint-best.pth \
    --checkpoint2 /HDD/data/yating/retfound_checkpoints/age/checkpoint-best.pth \
    --output_dir ./checkpoint_analysis
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from collections import OrderedDict
import pickle
from datetime import datetime

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze and compare RETFound checkpoints")
    parser.add_argument("--checkpoint1", type=str, required=True, 
                        help="Path to first checkpoint (VD model)")
    parser.add_argument("--checkpoint2", type=str, required=True,
                        help="Path to second checkpoint (Age model)")
    parser.add_argument("--output_dir", type=str, default="./checkpoint_analysis",
                        help="Output directory for analysis results")
    return parser.parse_args()

def safe_tensor_info(tensor):
    """Safely extract tensor information"""
    try:
        if isinstance(tensor, torch.Tensor):
            return {
                'shape': list(tensor.shape),
                'dtype': str(tensor.dtype),
                'device': str(tensor.device),
                'requires_grad': tensor.requires_grad,
                'mean': float(tensor.float().mean().item()) if tensor.numel() > 0 else 0.0,
                'std': float(tensor.float().std().item()) if tensor.numel() > 1 else 0.0,
                'min': float(tensor.min().item()) if tensor.numel() > 0 else 0.0,
                'max': float(tensor.max().item()) if tensor.numel() > 0 else 0.0,
                'num_elements': tensor.numel()
            }
        else:
            return {'type': str(type(tensor)), 'value': str(tensor)}
    except Exception as e:
        return {'error': f"Failed to process tensor: {str(e)}"}

def analyze_checkpoint(checkpoint_path, output_file):
    """Analyze a single checkpoint and save detailed information"""
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        analysis = {
            'checkpoint_path': checkpoint_path,
            'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
            'analysis_timestamp': datetime.now().isoformat(),
            'top_level_keys': list(checkpoint.keys()) if isinstance(checkpoint, dict) else ['root_tensor'],
            'checkpoint_structure': {}
        }
        
        # Analyze top-level structure
        if isinstance(checkpoint, dict):
            for key, value in checkpoint.items():
                print(f"  Processing key: {key}")
                
                if isinstance(value, torch.Tensor):
                    analysis['checkpoint_structure'][key] = {
                        'type': 'tensor',
                        'info': safe_tensor_info(value)
                    }
                elif isinstance(value, dict):
                    # This is likely the model state dict
                    analysis['checkpoint_structure'][key] = {
                        'type': 'state_dict',
                        'num_parameters': len(value),
                        'parameter_names': list(value.keys()),
                        'parameters': {}
                    }
                    
                    # Analyze each parameter
                    total_params = 0
                    for param_name, param_tensor in value.items():
                        param_info = safe_tensor_info(param_tensor)
                        analysis['checkpoint_structure'][key]['parameters'][param_name] = param_info
                        if 'num_elements' in param_info:
                            total_params += param_info['num_elements']
                    
                    analysis['checkpoint_structure'][key]['total_parameters'] = total_params
                    
                elif isinstance(value, (int, float, str, bool)):
                    analysis['checkpoint_structure'][key] = {
                        'type': 'scalar',
                        'value': value
                    }
                elif isinstance(value, list):
                    analysis['checkpoint_structure'][key] = {
                        'type': 'list',
                        'length': len(value),
                        'sample_items': value[:5] if len(value) > 0 else []
                    }
                else:
                    analysis['checkpoint_structure'][key] = {
                        'type': str(type(value)),
                        'info': str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
                    }
        
        # Save detailed analysis
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"  Analysis saved to: {output_file}")
        return analysis
        
    except Exception as e:
        error_analysis = {
            'checkpoint_path': checkpoint_path,
            'error': str(e),
            'analysis_timestamp': datetime.now().isoformat()
        }
        with open(output_file, 'w') as f:
            json.dump(error_analysis, f, indent=2)
        print(f"  Error analyzing checkpoint: {e}")
        return error_analysis

def compare_model_architectures(analysis1, analysis2, output_file):
    """Compare two checkpoint analyses and identify differences"""
    print("Comparing model architectures...")
    
    comparison = {
        'comparison_timestamp': datetime.now().isoformat(),
        'checkpoint1_path': analysis1.get('checkpoint_path', 'Unknown'),
        'checkpoint2_path': analysis2.get('checkpoint_path', 'Unknown'),
        'differences': {},
        'similarities': {},
        'architecture_analysis': {}
    }
    
    # Compare top-level keys
    keys1 = set(analysis1.get('top_level_keys', []))
    keys2 = set(analysis2.get('top_level_keys', []))
    
    comparison['differences']['top_level_keys'] = {
        'only_in_checkpoint1': list(keys1 - keys2),
        'only_in_checkpoint2': list(keys2 - keys1),
        'common_keys': list(keys1 & keys2)
    }
    
    # Find model state dicts
    model_key1 = None
    model_key2 = None
    
    for key in keys1:
        if analysis1['checkpoint_structure'].get(key, {}).get('type') == 'state_dict':
            model_key1 = key
            break
    
    for key in keys2:
        if analysis2['checkpoint_structure'].get(key, {}).get('type') == 'state_dict':
            model_key2 = key
            break
    
    if model_key1 and model_key2:
        print(f"  Found model state dicts: '{model_key1}' and '{model_key2}'")
        
        # Compare model parameters
        params1 = set(analysis1['checkpoint_structure'][model_key1]['parameter_names'])
        params2 = set(analysis2['checkpoint_structure'][model_key2]['parameter_names'])
        
        comparison['architecture_analysis'] = {
            'model_key1': model_key1,
            'model_key2': model_key2,
            'total_params1': analysis1['checkpoint_structure'][model_key1].get('total_parameters', 0),
            'total_params2': analysis2['checkpoint_structure'][model_key2].get('total_parameters', 0),
            'parameter_differences': {
                'only_in_model1': list(params1 - params2),
                'only_in_model2': list(params2 - params1),
                'common_parameters': list(params1 & params2)
            }
        }
        
        # Analyze parameter shape differences for common parameters
        shape_differences = {}
        common_params = params1 & params2
        
        for param_name in common_params:
            info1 = analysis1['checkpoint_structure'][model_key1]['parameters'][param_name]
            info2 = analysis2['checkpoint_structure'][model_key2]['parameters'][param_name]
            
            shape1 = info1.get('shape', [])
            shape2 = info2.get('shape', [])
            
            if shape1 != shape2:
                shape_differences[param_name] = {
                    'shape_model1': shape1,
                    'shape_model2': shape2
                }
        
        comparison['architecture_analysis']['shape_differences'] = shape_differences
        
        # Identify potential issues
        issues = []
        
        if len(params1 - params2) > 0:
            issues.append(f"Model 1 has {len(params1 - params2)} extra parameters")
        
        if len(params2 - params1) > 0:
            issues.append(f"Model 2 has {len(params2 - params1)} extra parameters")
        
        if shape_differences:
            issues.append(f"Found {len(shape_differences)} parameters with different shapes")
        
        # Check for head/classifier differences
        head_params1 = [p for p in params1 if any(head_name in p.lower() for head_name in ['head', 'classifier', 'fc', 'linear'])]
        head_params2 = [p for p in params2 if any(head_name in p.lower() for head_name in ['head', 'classifier', 'fc', 'linear'])]
        
        if set(head_params1) != set(head_params2):
            issues.append("Different head/classifier architecture detected")
            comparison['architecture_analysis']['head_analysis'] = {
                'head_params_model1': head_params1,
                'head_params_model2': head_params2
            }
        
        comparison['potential_issues'] = issues
        
    else:
        comparison['error'] = "Could not find model state dictionaries in one or both checkpoints"
    
    # Save comparison
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    print(f"  Comparison saved to: {output_file}")
    return comparison

def generate_summary_report(analysis1, analysis2, comparison, output_file):
    """Generate a human-readable summary report"""
    print("Generating summary report...")
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RETFound Checkpoint Architecture Analysis Summary\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Checkpoint information
        f.write("CHECKPOINT INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Checkpoint 1 (VD Model): {analysis1.get('checkpoint_path', 'Unknown')}\n")
        f.write(f"  File Size: {analysis1.get('file_size_mb', 0):.2f} MB\n")
        
        f.write(f"Checkpoint 2 (Age Model): {analysis2.get('checkpoint_path', 'Unknown')}\n")
        f.write(f"  File Size: {analysis2.get('file_size_mb', 0):.2f} MB\n\n")
        
        # Architecture comparison
        f.write("ARCHITECTURE COMPARISON:\n")
        f.write("-" * 40 + "\n")
        
        if 'architecture_analysis' in comparison:
            arch = comparison['architecture_analysis']
            f.write(f"Total Parameters Model 1: {arch.get('total_params1', 0):,}\n")
            f.write(f"Total Parameters Model 2: {arch.get('total_params2', 0):,}\n\n")
            
            param_diffs = arch.get('parameter_differences', {})
            f.write(f"Parameters only in Model 1: {len(param_diffs.get('only_in_model1', []))}\n")
            f.write(f"Parameters only in Model 2: {len(param_diffs.get('only_in_model2', []))}\n")
            f.write(f"Common parameters: {len(param_diffs.get('common_parameters', []))}\n\n")
            
            # List unique parameters
            if param_diffs.get('only_in_model1'):
                f.write("Parameters ONLY in VD Model:\n")
                for param in param_diffs['only_in_model1']:
                    f.write(f"  - {param}\n")
                f.write("\n")
            
            if param_diffs.get('only_in_model2'):
                f.write("Parameters ONLY in Age Model:\n")
                for param in param_diffs['only_in_model2']:
                    f.write(f"  - {param}\n")
                f.write("\n")
            
            # Shape differences
            shape_diffs = arch.get('shape_differences', {})
            if shape_diffs:
                f.write("SHAPE DIFFERENCES IN COMMON PARAMETERS:\n")
                f.write("-" * 40 + "\n")
                for param_name, shapes in shape_diffs.items():
                    f.write(f"{param_name}:\n")
                    f.write(f"  VD Model shape: {shapes['shape_model1']}\n")
                    f.write(f"  Age Model shape: {shapes['shape_model2']}\n\n")
            
            # Head analysis
            if 'head_analysis' in arch:
                f.write("HEAD/CLASSIFIER ANALYSIS:\n")
                f.write("-" * 40 + "\n")
                head_analysis = arch['head_analysis']
                f.write("VD Model head parameters:\n")
                for param in head_analysis['head_params_model1']:
                    f.write(f"  - {param}\n")
                f.write("\nAge Model head parameters:\n")
                for param in head_analysis['head_params_model2']:
                    f.write(f"  - {param}\n")
                f.write("\n")
        
        # Potential issues
        if 'potential_issues' in comparison:
            f.write("POTENTIAL ISSUES CAUSING SALIENCY MAP PROBLEMS:\n")
            f.write("-" * 50 + "\n")
            for i, issue in enumerate(comparison['potential_issues'], 1):
                f.write(f"{i}. {issue}\n")
            f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 40 + "\n")
        f.write("1. Check if both models use the same base architecture (same ViT configuration)\n")
        f.write("2. Verify that the LRP implementation is compatible with both model heads\n")
        f.write("3. If models have different output dimensions, ensure the regression LRP is configured correctly\n")
        f.write("4. Consider using the same model architecture with different trained weights\n")
        f.write("5. Check if the 'blue saliency' issue is related to output normalization or scaling\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"  Summary report saved to: {output_file}")

def main():
    """Main function"""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("RETFound Checkpoint Architecture Analyzer")
    print("=" * 50)
    
    # Analyze both checkpoints
    analysis1 = analyze_checkpoint(
        args.checkpoint1, 
        os.path.join(args.output_dir, "vd_model_analysis.json")
    )
    
    analysis2 = analyze_checkpoint(
        args.checkpoint2,
        os.path.join(args.output_dir, "age_model_analysis.json")
    )
    
    # Compare architectures
    comparison = compare_model_architectures(
        analysis1, 
        analysis2,
        os.path.join(args.output_dir, "architecture_comparison.json")
    )
    
    # Generate summary report
    generate_summary_report(
        analysis1,
        analysis2, 
        comparison,
        os.path.join(args.output_dir, "analysis_summary.txt")
    )
    
    print("\n" + "=" * 50)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_dir}")
    print("\nKey files generated:")
    print("- vd_model_analysis.json: Detailed VD model analysis") 
    print("- age_model_analysis.json: Detailed Age model analysis")
    print("- architecture_comparison.json: Detailed comparison")
    print("- analysis_summary.txt: Human-readable summary")
    
    # Print quick summary
    if 'potential_issues' in comparison and comparison['potential_issues']:
        print(f"\nPotential issues found: {len(comparison['potential_issues'])}")
        for issue in comparison['potential_issues']:
            print(f"  • {issue}")
    else:
        print("\nNo obvious architectural differences detected.")

if __name__ == "__main__":
    main()