import streamlit as st
from chatbot import generate_response

# ---------------------------------------
# PAGE CONFIG
# ---------------------------------------

st.set_page_config(
    page_title="AI Mobile Recommendation Chatbot",
    page_icon="📱",
    layout="wide"
)

# ---------------------------------------
# SIDEBAR
# ---------------------------------------

with st.sidebar:

    st.title("📱 AI Mobile Assistant")

    st.markdown("---")

    st.subheader("✨ Features")

    st.markdown("""
✅ Smart Phone Recommendations

✅ Detailed Phone Specifications

✅ Side-by-Side Comparisons

✅ Similar Phone Suggestions

✅ Conversation Memory
""")

    st.markdown("---")

    st.subheader("💬 Example Questions")

    st.markdown("""
🎮 Recommend a gaming phone under ₹40,000

📸 Best camera phone under ₹60,000

⚖️ Compare Samsung Galaxy S24 Ultra and iPhone 15 Pro

📱 Tell me about Google Pixel 9 Pro

🔍 Show phones similar to OnePlus 13
""")

    st.markdown("---")

    if st.button("🗑️ Clear Chat", width="stretch"):

        st.session_state.messages = [
            {
                "role": "assistant",
                "content":
                """👋 Welcome!

I'm your **AI Mobile Recommendation Assistant** 🤖

I can help you:

• 📱 Recommend smartphones
• 📖 Explain phone specifications
• 📊 Compare two phones
• 🔍 Find similar phones

Start by asking a question below!"""
            }
        ]

        st.rerun()

    st.markdown("---")

    st.caption("MADE WITH ❤️ BY KHAN RAZIUR REHMAN")

# ---------------------------------------
# TITLE
# ---------------------------------------

st.title("📱 AI Mobile Recommendation Chatbot")

st.caption("Your personal AI smartphone expert")

st.divider()

# ---------------------------------------
# CHAT HISTORY
# ---------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role":"assistant",
            "content":
"""👋 Hello!

I can help you with:

📱 Phone Recommendations

📖 Phone Specifications

📊 Phone Comparisons

🔍 Similar Phones

Try asking:

• Recommend a gaming phone under ₹40,000

• Compare Samsung Galaxy S24 Ultra and Google Pixel 9 Pro

• Tell me about iPhone 15"""
        }
    ]

# ---------------------------------------
# DISPLAY CHAT
# ---------------------------------------

for message in st.session_state.messages:

    avatar = "🤖" if message["role"]=="assistant" else "👤"

    with st.chat_message(message["role"], avatar=avatar):

        st.markdown(message["content"])

# ---------------------------------------
# USER INPUT
# ---------------------------------------

prompt = st.chat_input("Ask anything about smartphones...")

if prompt:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate response
    with st.spinner("🤖 Thinking..."):
        response = generate_response(prompt)

    # Display assistant response
    with st.chat_message("assistant", avatar="🤖"):

        if isinstance(response, str):
            st.markdown(response)
        else:
            st.dataframe(
                response,
                width="stretch"
            )

    # Save response to history
    if isinstance(response, str):
        history_response = response
    else:
        history_response = response.to_string(index=False)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": history_response
        }
    )