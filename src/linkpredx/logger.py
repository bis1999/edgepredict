"""Centralised logger config so every module gets consistent output."""
import logging

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s | %(name)s: %(message)s",
    level=logging.INFO,
)

get_logger = logging.getLogger  # convenience alias