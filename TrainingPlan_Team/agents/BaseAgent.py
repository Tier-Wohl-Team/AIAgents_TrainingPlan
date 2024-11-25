from abc import ABCMeta
import dotenv
dotenv.load_dotenv("../../.env")
dotenv.load_dotenv("TrainingPlan_Team/.env")

class BaseAgent(metaclass=ABCMeta):
    """Base class for static agent implementations."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Ensure `NAME` is defined in the class
        if not hasattr(cls, "NAME") or not isinstance(cls.NAME, str):
            raise TypeError(f"Class {cls.__name__} must define a class-level string 'NAME'.")

        # Ensure `action` is defined as a static method
        if "action" not in cls.__dict__ or not isinstance(cls.__dict__["action"], staticmethod):
            raise TypeError(f"Class {cls.__name__} must define a static method 'action'.")

    @classmethod
    def greetings(cls):
        print(f"Hello, I am the {cls.NAME} agent!")

