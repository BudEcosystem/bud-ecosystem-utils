"""
Utils for handling node registration in Bud.

This module provides utilities to assist with registering nodes in a Dapr state store.
"""

import logging
import json
import os
from dapr.clients import DaprClient
from dapr.clients.grpc._state import StateOptions, Concurrency, Consistency

# Set up logging
from bud_ecosystem_utils.logger import setup_logger

logger = setup_logger(__name__, logging.DEBUG)

# Constants for state store and service registry
STORE_NAME = os.getenv("BUD_STATE_STORE", "bud-state-store")
SERVICE_REGISTRY_KEY = os.getenv("BUD_SERVICE_REGISTRY_KEY", "service-registry")

EVENT_PUBSUB_NAME = os.getenv("EVENT_PUBSUB_NAME", "bud-redis-queue")
def store_node(client: DaprClient, node_info: dict) -> None:
    """
    Save node information to a Dapr state store.

    Args:
        client (DaprClient): The Dapr client instance.
        node_info (str): A dictionary containing the node information. 
                          It should have keys: 'name', 'type', 'job_topic'.

    Raises:
        KeyError: If any of the required keys in node_info are missing.
    """
    try:

        client.save_state(
            store_name=STORE_NAME,
            key=node_info.get('topic'),
            value=json.dumps(node_info),
            state_metadata={"contentType": "application/json"},
        )

        logger.info(f"State for {node_info.get('topic')} initialized successfully.")

    except KeyError:
        logger.error("Required keys missing in node_info.")
        raise Exception("Required keys missing in node_info.")

    except Exception as e:
        logger.error(f"Error in store_node: {str(e)}")
        raise e


def update_service_registry(client: DaprClient, job_topic: str) -> None:
    """
    Update the service registry in the Dapr state store with the given job_topic.

    Args:
        client (DaprClient): The Dapr client instance.
        job_topic (str): The topic related to the job.

    """
    while True:
        try:
            service_registry_state = client.get_state(
                store_name=STORE_NAME,
                key=SERVICE_REGISTRY_KEY,
            )

            keys_data = service_registry_state.data or b"[]"
            keys = json.loads(keys_data.decode("utf-8"))
            etag = service_registry_state.etag

            logger.info(f"Current Keys: {keys}")

            if job_topic not in keys:
                keys.append(job_topic)

            client.save_state(
                store_name=STORE_NAME,
                key=SERVICE_REGISTRY_KEY,
                value=json.dumps(keys),
                etag=etag,
                options=StateOptions(
                    concurrency=Concurrency.first_write,
                    consistency=Consistency.strong,
                ),
            )
            logger.info("State Updated For Registries")
            break

        except Exception as e:
            logger.error(f"Error in update_service_registry: {str(e)}")
            continue


def publish_node_registry_event(client: DaprClient, node_info: dict) -> None:
    try:
        client.publish_event(
            pubsub_name=EVENT_PUBSUB_NAME,
            topic_name="new-node-added",
            data=json.dumps(node_info),
            data_content_type="application/json",
        )
    except Exception as e:
        logger.error(f"Error in Registry Event: {e}")
        raise e


def register_node(client: DaprClient, node_info: dict) -> None:
    """
    Register a node in the Dapr state store.

    Args:
        client (DaprClient): The Dapr client instance.
        node_info (json): A dictionary containing the node information. 
                          It should have keys: 'name', 'type', 'job_topic'.

    Raises:
        KeyError: If any of the required keys in node_info are missing.
    """
    try:

        job_topic = node_info.get("topic")
        store_node(client, node_info)
        update_service_registry(client, job_topic)

        # Publish Node
        publish_node_registry_event(client,node_info)

    except KeyError:
        logger.error("Required keys missing in node_info.")
        raise Exception("Required keys missing in node_info.")
    except Exception as e:
        logger.error(f"Error in register_node: {str(e)}")
        raise e
