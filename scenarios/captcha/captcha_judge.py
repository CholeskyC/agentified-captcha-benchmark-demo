import argparse
import contextlib
import uvicorn
import asyncio
import logging
import os
import base64
import random
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    TaskState,
    Part,
    TextPart,
    FilePart,
    FileWithBytes,
)
from a2a.utils import (
    new_agent_text_message
)

from agentbeats.green_executor import GreenAgent, GreenExecutor
from agentbeats.models import EvalRequest, EvalResult
from agentbeats.tool_provider import ToolProvider
from agentbeats.client import create_message, send_message

from captcha_judge_common import CaptchaEval, CaptchaAttempt, captcha_judge_agent_card


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("captcha_judge")


class CaptchaJudge(GreenAgent):
    def __init__(self):
        self._required_roles = ["captcha_solver"]
        self._required_config_keys = ["dataset_path"]
        self._tool_provider = ToolProvider()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self._required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"
        missing_config_keys = set(self._required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Validate dataset path exists
        dataset_path = Path(request.config["dataset_path"])
        if not dataset_path.exists():
            return False, f"Dataset path does not exist: {dataset_path}"

        # Validate num_samples if provided
        if "num_samples" in request.config:
            try:
                num_samples = int(request.config["num_samples"])
                if num_samples <= 0:
                    return False, "num_samples must be positive"
            except Exception as e:
                return False, f"Can't parse num_samples: {e}"

        return True, "ok"

    async def run_eval(self, req: EvalRequest, updater: TaskUpdater) -> None:
        logger.info(f"Starting CAPTCHA evaluation: {req}")

        try:
            dataset_path = Path(req.config["dataset_path"])
            num_samples = int(req.config.get("num_samples", -1))  # -1 means all

            # Load CAPTCHA images
            captcha_files = list(dataset_path.glob("*.png"))
            if not captcha_files:
                await updater.failed(new_agent_text_message(f"No PNG files found in {dataset_path}"))
                return

            # Select samples
            if num_samples > 0:
                captcha_files = random.sample(captcha_files, min(num_samples, len(captcha_files)))

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Loaded {len(captcha_files)} CAPTCHA images for evaluation")
            )
            logger.info(f"Testing {len(captcha_files)} CAPTCHAs")

            # Evaluate each CAPTCHA
            attempts = []
            solver_url = str(req.participants["captcha_solver"])

            for i, captcha_file in enumerate(captcha_files, 1):
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Testing CAPTCHA {i}/{len(captcha_files)}: {captcha_file.name}")
                )

                attempt = await self.evaluate_captcha(captcha_file, solver_url)
                attempts.append(attempt)

                status = "✓ CORRECT" if attempt.correct else "✗ WRONG"
                logger.info(f"{status}: {captcha_file.name} - Predicted: '{attempt.predicted_text}', Actual: '{attempt.actual_text}'")

                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"{status} ({i}/{len(captcha_files)}): '{attempt.predicted_text}' vs '{attempt.actual_text}'"
                    )
                )

                # Rate limit: 10 requests per minute = 6 seconds between requests
                if i < len(captcha_files):
                    await asyncio.sleep(6)
                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(f"Waiting 6s for rate limit... ({i}/{len(captcha_files)} completed)")
                    )

            # Calculate metrics
            correct_count = sum(1 for a in attempts if a.correct)
            overall_accuracy = (correct_count / len(attempts)) * 100 if attempts else 0
            avg_char_accuracy = sum(a.character_accuracy for a in attempts) / len(attempts) if attempts else 0

            eval_result = CaptchaEval(
                total_attempts=len(attempts),
                correct_predictions=correct_count,
                overall_accuracy=overall_accuracy,
                average_character_accuracy=avg_char_accuracy,
                attempts=attempts
            )

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Evaluation complete! Accuracy: {overall_accuracy:.1f}% ({correct_count}/{len(attempts)})"
                )
            )
            logger.info(f"CAPTCHA Evaluation:\n{eval_result.model_dump_json(indent=2)}")

            # Create result artifact
            result = EvalResult(
                winner="captcha_solver" if overall_accuracy >= 50 else "baseline",
                detail=eval_result.model_dump()
            )

            # Create detailed summary
            summary = f"""CAPTCHA Benchmark Results
===========================
Total Attempts: {eval_result.total_attempts}
Correct: {eval_result.correct_predictions}
Overall Accuracy: {eval_result.overall_accuracy:.2f}%
Average Character Accuracy: {eval_result.average_character_accuracy:.2f}%

Detailed Results:
"""
            for attempt in eval_result.attempts:
                status_icon = "✓" if attempt.correct else "✗"
                summary += f"\n{status_icon} {attempt.filename}: '{attempt.predicted_text}' vs '{attempt.actual_text}' (char acc: {attempt.character_accuracy:.0f}%)"

            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=summary)),
                    Part(root=TextPart(text=result.model_dump_json(indent=2))),
                ],
                name="Result",
            )
        finally:
            self._tool_provider.reset()

    async def evaluate_captcha(self, captcha_file: Path, solver_url: str) -> CaptchaAttempt:
        """Evaluate a single CAPTCHA image."""
        # Extract ground truth from filename (without .png extension)
        actual_text = captcha_file.stem.lower()

        # Load and encode image
        with open(captcha_file, "rb") as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode("utf-8")

        # Create message with image
        prompt_text = "Extract the text from this CAPTCHA image. Return only the 5 characters you see, nothing else."
        message = create_message(
            text=prompt_text,
            context_id=None
        )

        # Add image part to the message
        message.parts.append(Part(root=FilePart(
            file=FileWithBytes(
                bytes=image_base64,
                mimeType="image/png",
                name=captcha_file.name
            )
        )))
        
        logger.info(f"Sending to solver: prompt='{prompt_text}', image={captcha_file.name}")

        # Send to solver using A2A client directly
        try:
            import httpx
            from a2a.client import A2ACardResolver, ClientConfig, ClientFactory

            async with httpx.AsyncClient(timeout=300) as httpx_client:
                resolver = A2ACardResolver(httpx_client=httpx_client, base_url=solver_url)
                agent_card = await resolver.get_agent_card()
                config = ClientConfig(httpx_client=httpx_client, streaming=False)
                factory = ClientFactory(config)
                client = factory.create(agent_card)

                predicted_text = ""
                async for event in client.send_message(message):
                    if isinstance(event, tuple) and len(event) == 2:
                        task, status_event = event
                        # Check if task has artifacts with the response
                        if task and hasattr(task, 'artifacts') and task.artifacts:
                            from agentbeats.client import merge_parts
                            # Extract text from the first artifact
                            for artifact in task.artifacts:
                                if hasattr(artifact, 'parts') and artifact.parts:
                                    predicted_text = merge_parts(artifact.parts).strip().lower()
                                    logger.info(f"Received from solver (artifacts): '{predicted_text}'")
                                    break
                        # Also check task.status.message as fallback
                        elif task and hasattr(task, 'status') and task.status.message:
                            from agentbeats.client import merge_parts
                            predicted_text = merge_parts(task.status.message.parts).strip().lower()
                            logger.info(f"Received from solver (message): '{predicted_text}'")
                    elif hasattr(event, 'parts'):
                        # Direct message response
                        from agentbeats.client import merge_parts
                        predicted_text = merge_parts(event.parts).strip().lower()
                        logger.info(f"Received from solver (direct): '{predicted_text}'")

            # Handle empty or invalid responses
            if not predicted_text:
                predicted_text = ""
            # Take only first 5 characters if longer
            predicted_text = predicted_text[:5]

        except Exception as e:
            logger.error(f"Error getting prediction for {captcha_file.name}: {e}")
            predicted_text = ""

        # Calculate correctness
        correct = (predicted_text == actual_text)

        # Calculate character-level accuracy
        char_accuracy = self._calculate_character_accuracy(predicted_text, actual_text)

        return CaptchaAttempt(
            filename=captcha_file.name,
            actual_text=actual_text,
            predicted_text=predicted_text,
            correct=correct,
            character_accuracy=char_accuracy
        )

    def _calculate_character_accuracy(self, predicted: str, actual: str) -> float:
        """Calculate percentage of correct characters."""
        if not actual:
            return 0.0

        # Pad shorter string with empty chars for comparison
        max_len = max(len(predicted), len(actual))
        predicted_padded = predicted.ljust(max_len)
        actual_padded = actual.ljust(max_len)

        correct_chars = sum(1 for p, a in zip(predicted_padded, actual_padded) if p == a)
        return (correct_chars / len(actual)) * 100


async def main():
    parser = argparse.ArgumentParser(description="Run the A2A CAPTCHA judge.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="External URL to provide in the agent card")
    parser.add_argument("--cloudflare-quick-tunnel", action="store_true", help="Use a Cloudflare quick tunnel. Requires cloudflared. This will override --card-url")
    args = parser.parse_args()

    if args.cloudflare_quick_tunnel:
        from agentbeats.cloudflare import quick_tunnel
        agent_url_cm = quick_tunnel(f"http://{args.host}:{args.port}")
    else:
        agent_url_cm = contextlib.nullcontext(args.card_url or f"http://{args.host}:{args.port}/")

    async with agent_url_cm as agent_url:
        agent = CaptchaJudge()
        executor = GreenExecutor(agent)
        agent_card = captcha_judge_agent_card("CaptchaJudge", agent_url)

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

        uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()

if __name__ == '__main__':
    asyncio.run(main())
