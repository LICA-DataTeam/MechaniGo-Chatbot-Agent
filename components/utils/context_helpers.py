from components.schemas import User, MechaniGoContext

def merge_user_memory(
    context: MechaniGoContext,
    payload: dict
) -> None:
    if not context or not getattr(context, "user_ctx", None):
        return
    
    user: User = context.user_ctx.user_memory or User()
    updated = user.model_copy(
        update=payload,
        deep=True
    )
    context.user_ctx.user_memory = updated