import os
os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

from fiftyone.core.utils import add_sys_path
import fiftyone.operators as foo
from fiftyone.operators import types

with add_sys_path(os.path.dirname(os.path.abspath(__file__))):
    from janus_vqa import process_dataset

def _handle_calling(
        uri, 
        sample_collection, 
        model_path,
        question,
        question_field,
        answer_field,
        delegate=False,
        **kwargs
        ):
    ctx = dict(dataset=sample_collection)

    params = dict(
        model_path=model_path,
        question=question,
        question_field=question_field,
        answer_field=answer_field,
        delegate=delegate,
        **kwargs
        )
    return foo.execute_operator(uri, ctx, params=params)

# Define available models
JANUS_MODELS = {
    "deepseek-ai/Janus-Pro-7B": "Janus Pro 7B",
    "deepseek-ai/Janus-Pro-1B": "Janus Base 1B",
}

class JanusVQAOperator(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="janus_vqa",
            label="Run Janus VL-GPT",
            description="Run the Janus VL-GPT model on your Dataset!",
            dynamic=True,
            icon="/assets/whale-svgrepo-com.svg",  # You'll need to provide an icon
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        # Model selection dropdown
        model_dropdown = types.Dropdown(label="Which Janus model would you like to use?")
        for k, v in JANUS_MODELS.items():
            model_dropdown.add_choice(k, label=v)

        inputs.enum(
            "model_path",
            values=model_dropdown.values(),
            label="Model",
            default="Janus-Pro-1B",
            description="Select from one of the available Janus models",
            view=model_dropdown,
            required=True
        )

        # Question input
        inputs.str(
            "question",
            label="Question",
            description="What question would you like to ask about each image?",
            required=True,
        )

        # Field names for storing results
        inputs.str(
            "question_field",            
            required=True,
            default="janus_question",
            label="Question Field",
            description="Name of the field to store the question in"
        )

        inputs.str(
            "answer_field",            
            required=True,
            default="janus_answer",
            label="Answer Field",
            description="Name of the field to store the model's answers in"
        )
        
        # Delegation option
        inputs.bool(
            "delegate",
            default=False,
            required=True,
            label="Delegate execution?",
            description=("If you choose to delegate this operation you must first have a delegated service running. "
            "You can launch a delegated service by running `fiftyone delegated launch` in your terminal"),
            view=types.CheckboxView(),
        )

        inputs.view_target(ctx)
        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        view = ctx.target_view()
        model_path = ctx.params.get("model_path")
        question = ctx.params.get("question")
        question_field = ctx.params.get("question_field")
        answer_field = ctx.params.get("answer_field")

        process_dataset(
            dataset=view,
            model_path=model_path,
            question=question,
            question_field=question_field,
            answer_field=answer_field,
        )
        
        ctx.ops.reload_dataset()

    def __call__(
            self, 
            sample_collection, 
            model_path,
            question,
            question_field,
            answer_field,
            delegate,
            **kwargs
            ):
        return _handle_calling(
            self.uri,
            sample_collection,
            model_path,
            question,
            question_field,
            answer_field,
            delegate,
            **kwargs
            )

def register(p):
    p.register(JanusVQAOperator)