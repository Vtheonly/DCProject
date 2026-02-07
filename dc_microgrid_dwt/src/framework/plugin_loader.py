"""
Plugin Loader Module - Industrial DC Microgrid Platform

Discovers and loads plugins from the plugins directory at runtime.
Supports dynamic agent registration and lifecycle management.
"""
import os
import sys
import importlib
import importlib.util
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.domain.interfaces import IPlugin, IAgent

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Dynamic plugin loader for extending the system at runtime.
    
    Plugins are discovered from the plugins/ directory and must
    implement the IPlugin interface.
    """
    
    def __init__(self, plugins_dir: str = None):
        if plugins_dir is None:
            # Default to project-level plugins directory
            base_path = Path(__file__).parent.parent.parent
            plugins_dir = str(base_path / "plugins")
        
        self.plugins_dir = plugins_dir
        self.loaded_plugins: Dict[str, IPlugin] = {}
        self.plugin_agents: Dict[str, List[IAgent]] = {}
        self._bus = None
        self._config = {}
        
    def set_bus(self, bus: Any):
        """Set the event bus for plugin initialization."""
        self._bus = bus
        
    def set_config(self, config: Dict[str, Any]):
        """Set global config for plugin initialization."""
        self._config = config

    def discover_plugins(self) -> List[str]:
        """
        Discover available plugins in the plugins directory.
        
        Returns:
            List of plugin directory names found.
        """
        plugins = []
        
        if not os.path.exists(self.plugins_dir):
            logger.warning(f"Plugins directory not found: {self.plugins_dir}")
            return plugins
            
        for item in os.listdir(self.plugins_dir):
            item_path = os.path.join(self.plugins_dir, item)
            
            # Check if it's a directory with __init__.py
            if os.path.isdir(item_path):
                init_file = os.path.join(item_path, "__init__.py")
                if os.path.exists(init_file):
                    plugins.append(item)
                    
        logger.info(f"Discovered {len(plugins)} plugins: {plugins}")
        return plugins

    def load_plugin(self, plugin_name: str) -> Optional[IPlugin]:
        """
        Load a single plugin by name.
        
        Args:
            plugin_name: Name of the plugin directory
            
        Returns:
            Loaded plugin instance or None if failed
        """
        if plugin_name in self.loaded_plugins:
            logger.warning(f"Plugin {plugin_name} already loaded")
            return self.loaded_plugins[plugin_name]
            
        plugin_path = os.path.join(self.plugins_dir, plugin_name)
        
        if not os.path.exists(plugin_path):
            logger.error(f"Plugin path not found: {plugin_path}")
            return None
            
        try:
            # Add plugin directory to path temporarily
            if self.plugins_dir not in sys.path:
                sys.path.insert(0, self.plugins_dir)
            
            # Import the plugin module
            module = importlib.import_module(plugin_name)
            
            # Look for Plugin class that implements IPlugin
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, IPlugin) and 
                    attr is not IPlugin):
                    plugin_class = attr
                    break
                    
            if plugin_class is None:
                logger.error(f"No IPlugin implementation found in {plugin_name}")
                return None
                
            # Instantiate and initialize
            plugin_instance = plugin_class()
            
            if self._bus is not None:
                plugin_config = self._config.get("plugins", {}).get(plugin_name, {})
                success = plugin_instance.initialize(self._bus, plugin_config)
                
                if not success:
                    logger.error(f"Plugin {plugin_name} initialization failed")
                    return None
                    
            self.loaded_plugins[plugin_name] = plugin_instance
            self.plugin_agents[plugin_name] = plugin_instance.get_agents()
            
            logger.info(f"Loaded plugin: {plugin_name} v{plugin_instance.version}")
            return plugin_instance
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}", exc_info=True)
            return None

    def load_all_plugins(self) -> Dict[str, IPlugin]:
        """
        Discover and load all available plugins.
        
        Returns:
            Dictionary of loaded plugins
        """
        discovered = self.discover_plugins()
        
        for plugin_name in discovered:
            self.load_plugin(plugin_name)
            
        return self.loaded_plugins

    def unload_plugin(self, plugin_name: str) -> bool:
        """
        Unload a plugin and cleanup its resources.
        
        Args:
            plugin_name: Name of the plugin to unload
            
        Returns:
            True if unloaded successfully
        """
        if plugin_name not in self.loaded_plugins:
            logger.warning(f"Plugin {plugin_name} not loaded")
            return False
            
        try:
            plugin = self.loaded_plugins[plugin_name]
            
            # Stop all agents from this plugin
            for agent in self.plugin_agents.get(plugin_name, []):
                try:
                    agent.stop()
                except Exception as e:
                    logger.error(f"Error stopping agent: {e}")
            
            # Shutdown plugin
            plugin.shutdown()
            
            # Remove from registries
            del self.loaded_plugins[plugin_name]
            del self.plugin_agents[plugin_name]
            
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False

    def unload_all_plugins(self):
        """Unload all loaded plugins."""
        plugin_names = list(self.loaded_plugins.keys())
        for name in plugin_names:
            self.unload_plugin(name)

    def get_all_agents(self) -> List[IAgent]:
        """Get all agents from all loaded plugins."""
        agents = []
        for plugin_agents in self.plugin_agents.values():
            agents.extend(plugin_agents)
        return agents

    def get_plugin(self, plugin_name: str) -> Optional[IPlugin]:
        """Get a loaded plugin by name."""
        return self.loaded_plugins.get(plugin_name)

    def get_plugin_info(self) -> List[Dict[str, str]]:
        """Get information about all loaded plugins."""
        return [
            {
                "name": plugin.name,
                "version": plugin.version,
                "description": plugin.description
            }
            for plugin in self.loaded_plugins.values()
        ]
