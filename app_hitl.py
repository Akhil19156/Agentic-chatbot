from agentic_chatbot_hitl_backend import (

    chatbot,

    get_all_threads,

    ingest_rag_document

)



from langchain_core.messages import (

    BaseMessage,

    HumanMessage,

    AIMessage,

    ToolMessage

)



from langgraph.types import Command



import streamlit as st

import uuid

import tempfile

import os
import json
from pathlib import Path





# ========================= Chat title helpers =========================

TITLES_FILE = Path(__file__).with_name("chat_titles.json")


def load_chat_titles():
    """Load saved chat titles from the local JSON file."""
    if TITLES_FILE.exists():
        try:
            return json.loads(TITLES_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_chat_titles():
    """Persist chat titles so they survive Streamlit reruns/restarts."""
    TITLES_FILE.write_text(
        json.dumps(st.session_state["chat_titles"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def create_chat_title(text):
    """Create a short title from the first user message."""
    text = " ".join(str(text).strip().split())
    if not text:
        return "New Chat"
    return text[:35] + ("..." if len(text) > 35 else "")

HIDDEN_CHATS_FILE = Path(__file__).with_name("hidden_chats.json")


def load_hidden_chats():
    """Load chat IDs hidden from the sidebar; this does not delete backend data."""
    try:
        if HIDDEN_CHATS_FILE.exists():
            data = json.loads(HIDDEN_CHATS_FILE.read_text(encoding="utf-8"))
            return set(data if isinstance(data, list) else [])
    except (OSError, json.JSONDecodeError):
        pass
    return set()


def save_hidden_chats():
    HIDDEN_CHATS_FILE.write_text(
        json.dumps(sorted(st.session_state["hidden_chat_ids"]), indent=2),
        encoding="utf-8",
    )



# Generate a unique thread ID for each new conversation

def generate_thread_id():

    return str(uuid.uuid4())





# Add a new thread ID to the conversation list

def add_thread(thread_id):



    # Prevent the same thread from being added multiple times

    if thread_id not in st.session_state["chat_threads"]:

        st.session_state["chat_threads"].append(thread_id)





# Create a completely new chat conversation

def reset_chat():



    # Generate and assign a new thread ID

    st.session_state["thread_id"] = generate_thread_id()



    # Clear the current chat messages from the UI

    st.session_state["message_history"] = []



    # ========================= HITL ADDED =========================

    # Clear any pending human approval request

    st.session_state["pending_hitl"] = None

    # =============================================================



    # Add the new thread to the conversation list

    add_thread(st.session_state["thread_id"])





# Load a previous conversation from the LangGraph checkpointer

def load_conversation(thread_id):



    # Get the saved state for the selected thread

    state = chatbot.get_state(

        config={

            "configurable": {

                "thread_id": thread_id

            }

        }

    )



    # Return saved messages

    # Return an empty list if no messages are available

    return state.values.get("messages", [])





# ========================= HITL helper functions =========================



def get_pending_interrupt(thread_id):

    """

    Return the first unresolved LangGraph interrupt for a thread.



    Returns:

        The pending Interrupt object, or None.

    """



    config = {

        "configurable": {

            "thread_id": thread_id

        }

    }



    try:



        # Read the current checkpoint state

        state_snapshot = chatbot.get_state(config)



        # Some LangGraph versions expose interrupts directly

        direct_interrupts = getattr(

            state_snapshot,

            "interrupts",

            ()

        ) or ()



        if direct_interrupts:

            return direct_interrupts[0]



        # Other LangGraph versions store interrupts inside tasks

        tasks = getattr(

            state_snapshot,

            "tasks",

            ()

        ) or ()



        for task in tasks:



            task_interrupts = getattr(

                task,

                "interrupts",

                ()

            ) or ()



            if task_interrupts:

                return task_interrupts[0]



    except Exception:



        # A newly created thread may not have a checkpoint yet

        return None



    return None





def save_pending_interrupt(thread_id, interrupt_object):

    """

    Save the pending interrupt information inside Streamlit state.

    """



    st.session_state["pending_hitl"] = {

        "thread_id": thread_id,

        "prompt": str(interrupt_object.value)

    }





def sync_pending_interrupt(thread_id):

    """

    Synchronize Streamlit HITL state with the LangGraph checkpoint.



    This allows a pending approval request to reappear after:

    - a Streamlit rerun

    - a browser refresh

    - switching between conversations

    """



    pending_interrupt = get_pending_interrupt(thread_id)



    if pending_interrupt is not None:



        save_pending_interrupt(

            thread_id,

            pending_interrupt

        )



    else:



        current_pending = st.session_state.get(

            "pending_hitl"

        )



        if (

            current_pending is not None

            and current_pending.get("thread_id") == thread_id

        ):

            st.session_state["pending_hitl"] = None





def resume_hitl_execution(decision):

    """

    Resume an interrupted LangGraph execution.



    Args:

        decision:

            "yes" approves the stock purchase.

            "no" rejects the stock purchase.

    """



    pending_hitl = st.session_state.get(

        "pending_hitl"

    )



    if not pending_hitl:



        st.warning(

            "There is no pending action to approve or reject."

        )



        return



    # Get the thread that originally triggered the interrupt

    interrupted_thread_id = pending_hitl["thread_id"]



    # The same thread ID must be used when resuming

    resume_config = {

        "configurable": {

            "thread_id": interrupted_thread_id

        },

        "metadata": {

            "thread_id": interrupted_thread_id

        },

        "run_name": "hitl_resume_trace",

    }



    try:



        # Display the resumed response

        with st.chat_message("assistant"):



            status_holder = {

                "box": st.status(

                    "🔄 Resuming the requested action...",

                    expanded=True

                )

            }



            def resumed_ai_only_stream():



                # Resume the graph with the human decision

                for message_chunk, metadata in chatbot.stream(

                    Command(resume=decision),

                    config=resume_config,

                    stream_mode="messages",

                ):



                    # Update tool execution status

                    if isinstance(

                        message_chunk,

                        ToolMessage

                    ):



                        tool_name = getattr(

                            message_chunk,

                            "name",

                            "tool"

                        )



                        status_holder["box"].update(

                            label=f"🔧 Using `{tool_name}` …",

                            state="running",

                            expanded=True,

                        )



                    # Stream only assistant-generated text

                    if isinstance(

                        message_chunk,

                        AIMessage

                    ):



                        if message_chunk.content:

                            yield message_chunk.content



            # Display the streamed final answer

            resumed_ai_message = st.write_stream(

                resumed_ai_only_stream()

            )



            # Check whether another interrupt occurred

            next_interrupt = get_pending_interrupt(

                interrupted_thread_id

            )



            if next_interrupt is not None:



                save_pending_interrupt(

                    interrupted_thread_id,

                    next_interrupt

                )



                status_holder["box"].update(

                    label="⚠️ Another approval is required",

                    state="complete",

                    expanded=False

                )



            else:



                # No more pending approval

                st.session_state["pending_hitl"] = None



                status_holder["box"].update(

                    label="✅ Action completed",

                    state="complete",

                    expanded=False

                )



        # Save the assistant response in Streamlit UI history

        if resumed_ai_message:



            st.session_state["message_history"].append({

                "role": "assistant",

                "content": resumed_ai_message

            })



        # Rerun so the response appears in normal chat order

        st.rerun()



    except Exception as error:



        st.error(

            f"Could not resume the requested action: {error}"

        )





# ========================= Page configuration =========================



st.set_page_config(

    page_title="Agentic Chatbot",

    page_icon="🤖"

)



# Display the main application title

st.title("Agentic Chatbot with LangGraph")





# Create message_history when the app runs for the first time

if "message_history" not in st.session_state:

    st.session_state["message_history"] = []





# Create a thread ID when the app runs for the first time

if "thread_id" not in st.session_state:

    st.session_state["thread_id"] = generate_thread_id()





# Create a list for storing all conversation thread IDs

if "chat_threads" not in st.session_state:

    st.session_state["chat_threads"] = get_all_threads()

if "hidden_chat_ids" not in st.session_state:
    st.session_state["hidden_chat_ids"] = load_hidden_chats()

# Keep hidden conversations out of the sidebar without touching the backend database.
st.session_state["chat_threads"] = [
    thread_id for thread_id in st.session_state["chat_threads"]
    if thread_id not in st.session_state["hidden_chat_ids"]
]

# Load conversation titles separately from LangGraph's checkpoint database.
if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = load_chat_titles()

if "renaming_thread" not in st.session_state:
    st.session_state["renaming_thread"] = None

# Ensure all known threads have a display title.
for existing_thread_id in st.session_state["chat_threads"]:
    st.session_state["chat_titles"].setdefault(
        existing_thread_id, "New Chat"
    )





# ========================= HITL ADDED =========================



# Store the currently pending human approval request

if "pending_hitl" not in st.session_state:

    st.session_state["pending_hitl"] = None



# =============================================================





# Add the current thread to the conversation list

add_thread(st.session_state["thread_id"])





# ========================= HITL ADDED =========================



# Recover pending approval after page refresh or rerun

sync_pending_interrupt(

    st.session_state["thread_id"]

)



# =============================================================





# ========================= Sidebar: ChatGPT-style conversations =========================

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.35rem; }
    [data-testid="stSidebar"] button { border-radius: 0.55rem; }
    [data-testid="stSidebar"] [data-testid="stPopover"] button { min-height: 2.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("My Conversations")

if st.sidebar.button("＋  New Chat", key="new_chat_button", use_container_width=True):
    st.session_state["renaming_thread"] = None
    reset_chat()
    st.rerun()


def get_display_title(thread_id):
    saved_title = st.session_state["chat_titles"].get(thread_id)
    if saved_title and saved_title != "New Chat":
        return saved_title

    try:
        for saved_message in load_conversation(thread_id):
            if isinstance(saved_message, HumanMessage):
                content = saved_message.content
                if not isinstance(content, str):
                    content = str(content)
                title = create_chat_title(content)
                st.session_state["chat_titles"][thread_id] = title
                return title
    except Exception:
        pass
    return "New Chat"


def persist_chat_title(thread_id, title):
    cleaned = " ".join(title.strip().split())
    if cleaned:
        st.session_state["chat_titles"][thread_id] = cleaned[:60]
        save_chat_titles()


for thread_id in reversed(st.session_state["chat_threads"]):
    title = get_display_title(thread_id)
    is_selected = thread_id == st.session_state["thread_id"]

    with st.sidebar.container():
        title_col, menu_col = st.columns([6, 1], gap="small")

        with title_col:
            if st.button(
                title,
                key=f"open_chat_{thread_id}",
                help=title,
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state["thread_id"] = thread_id
                messages = load_conversation(thread_id)
                restored_messages = []
                for message in messages:
                    if isinstance(message, HumanMessage):
                        restored_messages.append({
                            "role": "user",
                            "content": message.content,
                        })
                    elif isinstance(message, AIMessage):
                        content = message.content
                        if isinstance(content, str) and content.strip():
                            restored_messages.append({
                                "role": "assistant",
                                "content": content,
                            })
                st.session_state["message_history"] = restored_messages
                sync_pending_interrupt(thread_id)
                st.rerun()

        with menu_col:
            with st.popover(
                "⋯",
                key=f"conversation_menu_{thread_id}",
                help="Conversation options",
                use_container_width=True,
            ):
                st.caption("Conversation options")

                with st.form(key=f"rename_form_{thread_id}"):
                    new_title = st.text_input(
                        "Rename conversation",
                        value=title,
                        max_chars=60,
                        key=f"rename_value_{thread_id}",
                        label_visibility="collapsed",
                        placeholder="Conversation name",
                    )
                    rename_submitted = st.form_submit_button(
                        "Rename",
                        use_container_width=True,
                    )

                if rename_submitted:
                    if new_title.strip():
                        persist_chat_title(thread_id, new_title)
                        st.rerun()
                    else:
                        st.warning("Enter a conversation name.")

                st.divider()
                if st.button(
                    "Hide from sidebar",
                    key=f"hide_chat_{thread_id}",
                    use_container_width=True,
                ):
                    st.session_state["hidden_chat_ids"].add(thread_id)
                    save_hidden_chats()
                    st.session_state["chat_threads"] = [
                        item for item in st.session_state["chat_threads"]
                        if item != thread_id
                    ]
                    if st.session_state["thread_id"] == thread_id:
                        st.session_state["pending_hitl"] = None
                        reset_chat()
                    st.rerun()


# Save titles recovered from older conversations.
save_chat_titles()


# ========================= Main chat interface =========================



# Display all messages from the currently selected conversation

for message in st.session_state["message_history"]:



    # Create either a user chat bubble or assistant chat bubble

    with st.chat_message(message["role"]):



        # Display the message content

        st.text(message["content"])





# ========================= HITL approval interface =========================



# Get the currently pending approval request

pending_hitl = st.session_state.get(

    "pending_hitl"

)



# Check whether the pending approval belongs to

# the currently selected conversation

current_thread_has_pending_hitl = (

    pending_hitl is not None

    and pending_hitl.get("thread_id")

    == st.session_state["thread_id"]

)





# Display approval controls

if current_thread_has_pending_hitl:



    st.warning(

        "🧑 Human approval required\n\n"

        f"{pending_hitl['prompt']}"

    )



    approve_column, reject_column = st.columns(2)



    # Approve button

    with approve_column:



        if st.button(

            "✅ Approve Purchase",

            key=f"approve_{st.session_state['thread_id']}",

            type="primary",

            use_container_width=True

        ):



            # Send "yes" back to interrupt()

            resume_hitl_execution("yes")



    # Reject button

    with reject_column:



        if st.button(

            "❌ Reject Purchase",

            key=f"reject_{st.session_state['thread_id']}",

            use_container_width=True

        ):



            # Send "no" back to interrupt()

            resume_hitl_execution("no")





# ========================= Fixed chat input with PDF upload =========================



# Keep st.chat_input directly in the main body.

# This keeps it fixed at the bottom of the screen.

#

# accept_file=True adds the attachment button inside the chat input.

# file_type=["pdf"] allows PDF files only.

submission = st.chat_input(

    "Type here",

    accept_file=True,

    file_type=["pdf"],



    # Disable input while waiting for human approval

    disabled=current_thread_has_pending_hitl

)





# Default user input value

user_input = None





# Process the submitted text and PDF

if submission:



    # Get the text entered by the user

    user_input = submission.text



    # Get the uploaded files

    # This is always a list when accept_file is enabled

    uploaded_files = submission.files



    # Process the uploaded PDF if one was attached

    if uploaded_files:



        uploaded_pdf = uploaded_files[0]



        # Store the temporary file path

        temporary_file_path = None



        try:



            # Save the uploaded PDF as a temporary local file

            with tempfile.NamedTemporaryFile(

                delete=False,

                suffix=".pdf"

            ) as temporary_file:



                temporary_file.write(

                    uploaded_pdf.getvalue()

                )



                temporary_file_path = temporary_file.name



            # Call the existing backend RAG ingestion function

            with st.spinner(

                f"Processing {uploaded_pdf.name}..."

            ):



                ingest_rag_document(

                    temporary_file_path

                )



            # Display PDF processing confirmation

            st.toast(

                f"{uploaded_pdf.name} processed successfully.",

                icon="✅"

            )



        except Exception as error:



            # Display PDF processing error

            st.error(

                f"PDF processing failed: {error}"

            )



        finally:



            # Delete the temporary PDF after indexing

            if (

                temporary_file_path

                and os.path.exists(temporary_file_path)

            ):

                os.remove(temporary_file_path)





# Run this block after the user submits a text message

if user_input:

    # Automatically name a new chat using its first user message.
    current_thread_id = st.session_state["thread_id"]
    current_title = st.session_state["chat_titles"].get(
        current_thread_id, "New Chat"
    )
    if current_title == "New Chat":
        st.session_state["chat_titles"][current_thread_id] = (
            create_chat_title(user_input)
        )
        save_chat_titles()




    # Save the user's message in Streamlit session state

    st.session_state["message_history"].append({

        "role": "user",

        "content": user_input

    })



    # Display the user's message in the chat interface

    with st.chat_message("user"):

        st.text(user_input)



    # Pass the current thread ID to LangGraph

    # LangGraph uses this ID to save and retrieve conversation memory

    CONFIG = {

        "configurable": {

            "thread_id": st.session_state["thread_id"]

        },

        "metadata": {

            "thread_id": st.session_state["thread_id"]

        },

        "run_name": "chat_trace",

    }



    # Assistant streaming block

    with st.chat_message("assistant"):



        # Use a mutable holder so the generator can set/modify it

        status_holder = {

            "box": None

        }



        def ai_only_stream():



            for message_chunk, metadata in chatbot.stream(

                {

                    "messages": [

                        HumanMessage(content=user_input)

                    ]

                },

                config=CONFIG,

                stream_mode="messages",

            ):



                # Lazily create & update the SAME status container

                # when any tool runs

                if isinstance(

                    message_chunk,

                    ToolMessage

                ):



                    tool_name = getattr(

                        message_chunk,

                        "name",

                        "tool"

                    )



                    if status_holder["box"] is None:



                        status_holder["box"] = st.status(

                            f"🔧 Using `{tool_name}` …",

                            expanded=True

                        )



                    else:



                        status_holder["box"].update(

                            label=f"🔧 Using `{tool_name}` …",

                            state="running",

                            expanded=True,

                        )



                # Stream ONLY assistant tokens

                if isinstance(message_chunk, AIMessage):
                    if isinstance(message_chunk.content, str) and message_chunk.content:
                        yield message_chunk.content



            # ========================= HITL ADDED =========================



            # interrupt() pauses the graph without returning

            # a completed ToolMessage.

            #

            # Inspect the saved checkpoint after streaming ends.

            pending_interrupt = get_pending_interrupt(

                st.session_state["thread_id"]

            )



            if pending_interrupt is not None:



                # Save the interrupt for displaying approval buttons

                save_pending_interrupt(

                    st.session_state["thread_id"],

                    pending_interrupt

                )



                yield (

                    "\n\n⚠️ This stock purchase requires your approval. "

                    "Use the Approve Purchase or Reject Purchase "

                    "button below."

                )



            # =============================================================



        ai_message = st.write_stream(

            ai_only_stream()

        )



        # Finalize only if a tool was actually used

        if status_holder["box"] is not None:



            # Check whether execution is waiting for approval

            if get_pending_interrupt(

                st.session_state["thread_id"]

            ) is not None:



                status_holder["box"].update(

                    label="⏸️ Waiting for human approval",

                    state="complete",

                    expanded=False

                )



            else:



                status_holder["box"].update(

                    label="✅ Tool finished",

                    state="complete",

                    expanded=False

                )



    # Save the complete assistant response in Streamlit session state

    st.session_state["message_history"].append({

        "role": "assistant",

        "content": ai_message

    })



    # ========================= HITL ADDED =========================



    # Approval controls are rendered earlier in the script.

    # Rerun so they appear immediately after interrupt().

    if (

        st.session_state.get("pending_hitl") is not None

        and st.session_state["pending_hitl"].get("thread_id")

        == st.session_state["thread_id"]

    ):

        st.rerun()



    # =============================================================