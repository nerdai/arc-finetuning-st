import logging
import streamlit as st
from typing import Optional

from llama_deploy.control_plane import ControlPlaneConfig

logger = logging.getLogger(__name__)


class Controller:
    def __init__(
        self, control_plane_config: Optional[ControlPlaneConfig] = None
    ) -> None:
        self.control_plane_config = control_plane_config or ControlPlaneConfig()

    def handle_selectbox_selection(self):
        """Handle selection of ARC task."""
        selected_task = st.session_state.get("selected_task")
        print(selected_task)
