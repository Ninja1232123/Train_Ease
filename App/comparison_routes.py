"""
Model Comparison Routes - MVP API Endpoints
Flask routes for side-by-side model comparison.
"""

import logging
import os
import time
import json
from typing import Tuple

from flask import Flask, request, jsonify, Response

from comparison_manager import ComparisonManager


def add_comparison_routes(
    app: Flask,
    comparison_manager: ComparisonManager,
    logger: logging.Logger
) -> None:
    """Register model comparison API routes.

    Args:
        app: Flask application instance
        comparison_manager: ComparisonManager instance
        logger: Logger for tracking operations
    """

    @app.route('/api/comparison/load-model-a', methods=['POST'])
    def load_comparison_model_a() -> Tuple[Response, int]:
        """Load first model for comparison.

        Expected JSON:
            {
                "model_path": "/path/to/model"
            }

        Returns:
            JSON response with success status and model info
        """
        try:
            data = request.json
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            model_path = data.get('model_path')
            if not model_path:
                return jsonify({'error': 'No model_path provided'}), 400

            logger.info(f"API request to load Model A: {model_path}")
            success = comparison_manager.load_model_a(model_path)

            if success:
                return jsonify({
                    'success': True,
                    'message': 'Model A loaded successfully',
                    'status': comparison_manager.get_comparison_status()
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to load Model A - check logs for details'
                }), 500

        except Exception as e:
            logger.error(f"Error loading comparison Model A: {e}", exc_info=True)
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/comparison/load-model-b', methods=['POST'])
    def load_comparison_model_b() -> Tuple[Response, int]:
        """Load second model for comparison.

        Expected JSON:
            {
                "model_path": "/path/to/model"
            }

        Returns:
            JSON response with success status and model info
        """
        try:
            data = request.json
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            model_path = data.get('model_path')
            if not model_path:
                return jsonify({'error': 'No model_path provided'}), 400

            logger.info(f"API request to load Model B: {model_path}")
            success = comparison_manager.load_model_b(model_path)

            if success:
                return jsonify({
                    'success': True,
                    'message': 'Model B loaded successfully',
                    'status': comparison_manager.get_comparison_status()
                }), 200
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to load Model B - check logs for details'
                }), 500

        except Exception as e:
            logger.error(f"Error loading comparison Model B: {e}", exc_info=True)
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/comparison/generate', methods=['POST'])
    def comparison_generate() -> Tuple[Response, int]:
        """Generate from both models and return comparison.

        Expected JSON:
            {
                "prompt": "Text prompt",
                "max_tokens": 100,      # optional
                "temperature": 0.7,     # optional
                "top_p": 0.9,          # optional
                "top_k": 50            # optional
            }

        Returns:
            JSON with outputs from both models and comparison metrics
        """
        try:
            data = request.json
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            prompt = data.get('prompt', '').strip()
            if not prompt:
                return jsonify({'error': 'Prompt cannot be empty'}), 400

            # Get generation parameters with defaults
            params = {
                'max_tokens': int(data.get('max_tokens', 100)),
                'temperature': float(data.get('temperature', 0.7)),
                'top_p': float(data.get('top_p', 0.9)),
                'top_k': int(data.get('top_k', 50))
            }

            logger.info(f"Comparison generation requested - params: {params}")

            # Generate comparison
            result = comparison_manager.generate_comparison(prompt, **params)

            if result.get('success'):
                logger.info("Comparison generation successful")
                return jsonify(result), 200
            else:
                logger.error(f"Comparison generation failed: {result.get('error')}")
                return jsonify(result), 500

        except ValueError as e:
            logger.error(f"Invalid parameter: {e}")
            return jsonify({'success': False, 'error': f'Invalid parameter: {str(e)}'}), 400
        except Exception as e:
            logger.error(f"Error in comparison generation: {e}", exc_info=True)
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/comparison/status', methods=['GET'])
    def comparison_status() -> Tuple[Response, int]:
        """Get current comparison status.

        Returns:
            JSON with loading status of both models
        """
        try:
            status = comparison_manager.get_comparison_status()
            return jsonify(status), 200
        except Exception as e:
            logger.error(f"Error getting comparison status: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/comparison/unload', methods=['POST'])
    def comparison_unload() -> Tuple[Response, int]:
        """Unload both comparison models.

        Returns:
            JSON success response
        """
        try:
            logger.info("Unloading all comparison models")
            comparison_manager.unload_all()
            return jsonify({
                'success': True,
                'message': 'All comparison models unloaded'
            }), 200
        except Exception as e:
            logger.error(f"Error unloading comparison models: {e}", exc_info=True)
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/comparison/unload-model-a', methods=['POST'])
    def comparison_unload_model_a() -> Tuple[Response, int]:
        """Unload Model A only.

        Returns:
            JSON success response
        """
        try:
            logger.info("Unloading comparison Model A")
            comparison_manager.unload_model_a()
            return jsonify({
                'success': True,
                'message': 'Model A unloaded',
                'status': comparison_manager.get_comparison_status()
            }), 200
        except Exception as e:
            logger.error(f"Error unloading Model A: {e}", exc_info=True)
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/comparison/unload-model-b', methods=['POST'])
    def comparison_unload_model_b() -> Tuple[Response, int]:
        """Unload Model B only.

        Returns:
            JSON success response
        """
        try:
            logger.info("Unloading comparison Model B")
            comparison_manager.unload_model_b()
            return jsonify({
                'success': True,
                'message': 'Model B unloaded',
                'status': comparison_manager.get_comparison_status()
            }), 200
        except Exception as e:
            logger.error(f"Error unloading Model B: {e}", exc_info=True)
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/api/comparison/save', methods=['POST'])
    def save_comparison() -> Tuple[Response, int]:
        """Save comparison results to file.

        Expected JSON:
            The full comparison result object

        Returns:
            JSON with saved file path and comparison ID
        """
        try:
            data = request.json
            if not data:
                return jsonify({'error': 'No data provided'}), 400

            # Generate comparison ID
            comparison_id = f"comparison_{int(time.time())}"

            # Create output directory
            output_dir = './outputs/comparisons'
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f"{comparison_id}.json")

            # Save to file
            with open(output_path, 'w') as f:
                json.dump({
                    'id': comparison_id,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'comparison': data
                }, f, indent=2)

            logger.info(f"Saved comparison to {output_path}")

            return jsonify({
                'success': True,
                'comparison_id': comparison_id,
                'saved_path': output_path
            }), 200

        except Exception as e:
            logger.error(f"Error saving comparison: {e}", exc_info=True)
            return jsonify({'success': False, 'error': str(e)}), 500

    logger.info("Model comparison routes registered")
