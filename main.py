import gc
import json
import os
from typing import Any, Dict

import yaml
from loguru import logger
import torch

from process import SequentialProcessor
from src.utilities.tools.retry import retry


class RuntimeEngine:
    def __init__(self):
        self.shared_data: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def add_shared_data(self, name: str, data: Any):
        """stores shared data for sequential processing"""
        self.shared_data[name] = data

    def get_shared_data(self, name: str) -> Any:
        return self.shared_data.get(name, {})

    def add_metadata(self, name: str, data: Any):
        """stores metadata"""
        self.metadata[name] = data

    def get_metadata(self, name: str) -> Any:
        return self.metadata.get(name)


def initialize_runtime_engine():
    """Initialize the runtime engine with config and device"""
    runtime_engine = RuntimeEngine()
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Add metadata
    runtime_engine.add_metadata("config", config)
    runtime_engine.add_metadata("device", device)
    
    logger.info(f"Runtime engine initialized with device: {device}")
    return runtime_engine


def main():
    """Main entry point"""
    try:
        # Initialize runtime engine
        runtime_engine = initialize_runtime_engine()
        
        # Create and run sequential processor
        processor = SequentialProcessor(runtime_engine)
        processor.run()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
