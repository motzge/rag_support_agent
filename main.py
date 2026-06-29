from core.agent import run_agent
from logger import get_logger



logger = get_logger(__name__)


def main() -> None:
    """Main CLI Loop"""
    print("Python Docs Assistant - type 'exit' to quit.")
    print("-" * 50)

    while True:
        user_input: str = input("\nYou: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        response: str = run_agent(user_input)
        print(f"\nAssistant: {response}")




if __name__ == "__main__":
    main()