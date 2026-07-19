# ==========================================================
# AI MOBILE RECOMMENDATION CHATBOT
# Backend Engine
# ==========================================================

# ==========================================================
# IMPORT LIBRARIES
# ==========================================================

from pathlib import Path
import re

import nltk
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ==========================================================
# NLTK SETUP
# ==========================================================

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ==========================================================
# LOAD DATASET
# ==========================================================

BASE_DIR = Path(__file__).parent

DATASET_PATH = BASE_DIR / "data" / "phone_dataset.csv"

df = pd.read_csv(DATASET_PATH)

# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9 ]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# ==========================================================
# SEARCH DESCRIPTION
# ==========================================================

df["description"] = (

    df["Name"].astype(str)

    + " "

    + df["Company"].astype(str)

    + " "

    + df["Processor"].astype(str)

    + " "

    + df["Price_Category"].astype(str)

)

df["description"] = df["description"].apply(clean_text)

# ==========================================================
# TF-IDF MODEL
# ==========================================================

vectorizer = TfidfVectorizer()

phone_vectors = vectorizer.fit_transform(

    df["description"]

)

# ==========================================================
# QUERY PROCESSING
# ==========================================================

def process_query(query):

    query = clean_text(query)

    return vectorizer.transform([query])

# ==========================================================
# SIMILAR PHONE DESCRIPTION
# ==========================================================

df["similarity_description"] = (

    df["Processor"].astype(str)

    + " "

    + df["Processor_Brand"].astype(str)

    + " "

    + df["RAM_GB"].astype(str)

    + "GB "

    + df["ROM_GB"].astype(str)

    + "GB "

    + df["Battery_mAh"].astype(str)

    + "mAh "

    + df["Display_Type"].astype(str)

    + " "

    + df["Refresh_Rate"].astype(str)

    + "Hz "

    + df["Rear_Camera_MP"].astype(str)

    + "MP"

)

df["similarity_description"] = (

    df["similarity_description"]

    .apply(clean_text)

)

# ==========================================================
# SIMILARITY MODEL
# ==========================================================

similarity_vectorizer = TfidfVectorizer()

similarity_vectors = similarity_vectorizer.fit_transform(

    df["similarity_description"]

)

# ==========================================================
# NUMERIC FEATURES
# ==========================================================

numeric_features = [

    "Processor_Score",

    "RAM_GB",

    "ROM_GB",

    "Battery_mAh",

    "Refresh_Rate",

    "Rear_Camera_MP",

    "Camera_Score",

    "Gaming_Score",

    "Performance_Score",

    "Battery_Score",

    "Recommendation_Score"

]

scaler = MinMaxScaler()

numeric_matrix = scaler.fit_transform(

    df[numeric_features]

)

# ==========================================================
# BRAND DICTIONARY
# ==========================================================

brand_map = {

    "apple": "Apple",
    "iphone": "Apple",

    "samsung": "Samsung",

    "oneplus": "OnePlus",

    "google": "Google",
    "pixel": "Google",

    "xiaomi": "Xiaomi",
    "redmi": "Xiaomi",
    "poco": "Xiaomi",

    "realme": "Realme",

    "vivo": "Vivo",

    "oppo": "Oppo",

    "motorola": "Motorola",

    "nothing": "Nothing",

    "iqoo": "iQOO",

    "nokia": "Nokia"

}

# ==========================================================
# FEATURE KEYWORDS
# ==========================================================

feature_words = {

    "gaming": [
        "gaming",
        "game",
        "pubg",
        "bgmi",
        "fps"
    ],

    "camera": [
        "camera",
        "photo",
        "photography",
        "pictures"
    ],

    "battery": [
        "battery",
        "backup",
        "long battery"
    ],

    "performance": [
        "performance",
        "processor",
        "speed",
        "fast"
    ]

}

# ==========================================================
# BUDGET EXTRACTION
# ==========================================================

def extract_budget(text):

    text = text.lower().replace(",", "")

    match = re.search(r"(\d+)\s*k", text)

    if match:

        return int(match.group(1)) * 1000

    numbers = re.findall(r"\d+", text)

    if numbers:

        return int(numbers[0])

    return None


# ==========================================================
# BRAND EXTRACTION
# ==========================================================

def extract_brand(text):

    text = text.lower()

    for key, brand in brand_map.items():

        if key in text:

            return brand

    return None


# ==========================================================
# FEATURE EXTRACTION
# ==========================================================

def extract_feature(text):

    text = text.lower()

    for feature, words in feature_words.items():

        for word in words:

            if word in text:

                return feature

    return None


# ==========================================================
# PHONE ALIAS DICTIONARY
# ==========================================================

phone_dictionary = {}

for phone in df["Name"]:

    full = phone.lower()

    phone_dictionary[full] = phone

    words = full.split()

    # Last two words
    if len(words) >= 2:

        phone_dictionary[" ".join(words[-2:])] = phone

    # Remove company name
    if words[0] in [

        "google",
        "samsung",
        "apple"

    ]:

        alias = " ".join(words[1:])

        phone_dictionary[alias] = phone

    # Remove Pro / Ultra / Plus

    if words[-1] in [

        "pro",
        "ultra",
        "plus"

    ]:

        alias = " ".join(words[:-1])

        phone_dictionary[alias] = phone


# ==========================================================
# PHONE NAME EXTRACTION
# ==========================================================

def extract_phone_names(user_input):

    text = user_input.lower()

    phones = []

    used_spans = []

    # Search longest aliases first
    aliases = sorted(
        phone_dictionary.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    for alias, phone in aliases:

        pattern = r"\b" + re.escape(alias) + r"\b"

        match = re.search(pattern, text)

        if not match:
            continue

        start, end = match.span()

        # Skip overlapping matches
        overlap = False

        for s, e in used_spans:

            if not (end <= s or start >= e):
                overlap = True
                break

        if overlap:
            continue

        phones.append(phone)
        used_spans.append((start, end))

    # Remove duplicates while preserving order
    phones = list(dict.fromkeys(phones))

    return phones

# ==========================================================
# INTENT DETECTION
# ==========================================================

def detect_intent(user_input):

    text = user_input.lower()

    # Greeting
    if re.search(r"\b(hello|hi|hey)\b", text):
        return "greeting"

    if "help" in text:

        return "help"

    if any(

        word in text

        for word in [

            "recommend",
            "suggest",
            "best"

        ]

    ):

        return "recommend"
    
    if any(

         word in text

        for word in [

        "compare",
        "comparison",
        "vs",
        "versus",
        "difference",
        "better"

    ]

):

     return "compare"

    if any(

        word in text

        for word in [

            "similar",
            "alternative"

        ]

    ):

        return "similar"

    # If a phone name exists,
    # treat it as specification request

    if len(extract_phone_names(text)):

        return "spec"

    if any(

        word in text

        for word in [

            "about",
            "details",
            "specification",
            "specifications"

        ]

    ):

        return "spec"

    return "unknown"


# ==========================================================
# USER PROFILE
# ==========================================================

def understand_user(text):

    return {

        "budget": extract_budget(text),

        "brand": extract_brand(text),

        "feature": extract_feature(text)

    }

def detect_luxury(text):

    text = text.lower()

    if "ultra luxury" in text:
        return "Ultra Luxury"

    if any(word in text for word in [
        "luxury",
        "premium",
        "flagship"
    ]):
        return "Luxury"

    return "Normal"

# ==========================================================
# QUERY UNDERSTANDING
# ==========================================================

def understand_query(user_input):

    return {

        "intent": detect_intent(user_input),

        "budget": extract_budget(user_input),

        "brand": extract_brand(user_input),

        "feature": extract_feature(user_input),

        "luxury": detect_luxury(user_input),

        "phones": extract_phone_names(user_input)

    }

# ==========================================================
# AI RECOMMENDATION ENGINE
# ==========================================================

def advanced_recommend(query, top_n=5):

    user = understand_user(query)

    result = df.copy()

    print("\n===== RECOMMEND DEBUG =====")
    print("Query:", query)
    print("User Profile:", user)
    print("Total phones:", len(result))

    # -----------------------------------------
    # Budget Filter
    # -----------------------------------------

    if user["budget"] is not None:

        result = result[
            result["Price"].astype(float)
            <= user["budget"]
        ]

        print("After budget filter:", len(result))

    # -----------------------------------------
    # Brand Filter
    # -----------------------------------------

    if user["brand"] is not None:

        result = result[
            result["Company"]
            .str.lower()
            .str.contains(
                user["brand"].lower(),
                na=False
            )
        ]

        print("After brand filter:", len(result))

    print("Final filtered phones:", len(result))
    if result.empty:

        return result

    # -----------------------------------------
    # TF-IDF Similarity
    # -----------------------------------------

    query_vector = process_query(query)

    vectors = vectorizer.transform(

        result["description"]

    )

    similarity = cosine_similarity(

        query_vector,

        vectors

    )[0]

    result["NLP_Score"] = similarity * 100

    # -----------------------------------------
    # Feature Score
    # -----------------------------------------

    feature = user["feature"]

    if feature == "gaming":

        result["Feature_Score"] = result["Gaming_Score"]

    elif feature == "camera":

        result["Feature_Score"] = result["Camera_Score"]

    elif feature == "battery":

        result["Feature_Score"] = result["Battery_Score"]

    elif feature == "performance":

        result["Feature_Score"] = result["Performance_Score"]

    else:

        result["Feature_Score"] = 70

    # -----------------------------------------
    # Final AI Score
    # -----------------------------------------

    result["Final_Score"] = (

        result["NLP_Score"] * 0.50

        +

        result["Feature_Score"] * 0.30

        +

        result["Star_Rating"] * 20

    )

    result = result.sort_values(

        "Final_Score",

        ascending=False

    )

    return result.head(top_n)


# ==========================================================
# AI EXPLANATION
# ==========================================================

def generate_ai_reason(phone, user):

    reasons = []

    if user["budget"]:

        if phone["Price"] <= user["budget"]:

            reasons.append(

                f"💰 Fits your budget (₹{user['budget']:,})"

            )

    if user["brand"]:

        if phone["Company"] == user["brand"]:

            reasons.append(

                f"🏢 Matches your preferred brand ({phone['Company']})"

            )

    if user["feature"] == "gaming":

        reasons.append(

            f"🎮 Gaming Score: {phone['Gaming_Score']}/100"

        )

    elif user["feature"] == "camera":

        reasons.append(

            f"📸 Camera Score: {phone['Camera_Score']}/100"

        )

    elif user["feature"] == "battery":

        reasons.append(

            f"🔋 Battery Score: {phone['Battery_Score']}/100"

        )

    elif user["feature"] == "performance":

        reasons.append(

            f"⚡ Performance Score: {phone['Performance_Score']}/100"

        )

    if phone["Star_Rating"] >= 4.5:

        reasons.append(

            f"⭐ Excellent user rating ({phone['Star_Rating']}/5)"

        )

    if phone["Value_For_Money_Score"] >= 90:

        reasons.append(

            "💎 Excellent value for money"

        )

    return reasons


# ==========================================================
# FORMAT RECOMMENDATIONS
# ==========================================================

def format_recommendations(result, user):

    if result is None or result.empty:

        return (

            "😔 Sorry, I couldn't find any smartphones "

            "matching your requirements."

        )

    response = "# 📱 Recommended Smartphones\n\n"

    for rank, (_, phone) in enumerate(

        result.iterrows(),

        start=1

    ):

        response += (

            f"## 🏆 Recommendation {rank}\n\n"

            f"**📱 {phone['Name']}**\n\n"

            f"🏢 **Brand:** {phone['Company']}\n\n"

            f"💰 **Price:** ₹{phone['Price']}\n\n"

            f"⚙ **Processor:** {phone['Processor']}\n\n"

            f"⭐ **Rating:** {phone['Star_Rating']}/5\n\n"

        )

        reasons = generate_ai_reason(

            phone,

            user

        )

        if reasons:

            response += "### Why I recommend it\n\n"

            for reason in reasons:

                response += f"✅ {reason}\n\n"

        response += "---\n\n"

    return response

# ==========================================================
# PHONE SEARCH ENGINE
# ==========================================================

def find_phone(phone_name):

    if phone_name is None:

        return None

    phone_name = phone_name.lower().strip()

    # Check alias dictionary first
    if phone_name in phone_dictionary:

        phone_name = phone_dictionary[phone_name]

    # Exact match
    exact = df[
        df["Name"].str.lower() == phone_name.lower()
    ]

    if not exact.empty:

        return exact.iloc[0]

    # Partial match
    partial = df[
        df["Name"].str.lower().str.contains(
            phone_name.lower(),
            na=False
        )
    ]

    if not partial.empty:

        return partial.iloc[0]

    return None


# ==========================================================
# PHONE SPECIFICATIONS
# ==========================================================

def get_phone_specification(phone_name):

    phone = find_phone(phone_name)

    if phone is None:

        return "❌ Phone not found."

    response = f"""
# 📱 {phone['Name']}

---

## 🏢 Brand
**{phone['Company']}**

## 💰 Price
**₹{phone['Price']}**

## ⚙ Processor
**{phone['Processor']}**

## 🧠 RAM
**{phone['RAM_GB']} GB**

## 💾 Storage
**{phone['ROM_GB']} GB**

## 🔋 Battery
**{phone['Battery_mAh']} mAh**

## ⚡ Charging
**{phone['Charging_Watt']} W**

## 📺 Display
**{phone['Display_Size']}" {phone['Display_Type']}**

**{phone['Refresh_Rate']} Hz Refresh Rate**

## 📷 Rear Camera
**{phone['Rear_Camera_MP']} MP**

## 🤳 Front Camera
**{phone['Front_Camera_MP']} MP**

## ⭐ User Rating
**{phone['Star_Rating']} / 5**

---

# 📊 AI Scores

🎮 **Gaming Score:** {phone['Gaming_Score']}/100

📸 **Camera Score:** {phone['Camera_Score']}/100

🚀 **Performance Score:** {phone['Performance_Score']}/100

🔋 **Battery Score:** {phone['Battery_Score']}/100

💎 **Value for Money:** {phone['Value_For_Money_Score']}/100

🏆 **Best For:** {phone['Best_For']}
"""

    return response


# ==========================================================
# FIND SIMILAR PHONES
# ==========================================================

def find_similar_phones(phone_name, top_n=5):

    phone = find_phone(phone_name)

    if phone is None:

        return None

    index = phone.name

    text_similarity = cosine_similarity(

        similarity_vectors[index],

        similarity_vectors

    )[0]

    numeric_similarity = cosine_similarity(

        numeric_matrix[index].reshape(1, -1),

        numeric_matrix

    )[0]

    final_similarity = (

        0.6 * text_similarity

        +

        0.4 * numeric_similarity

    )

    indices = final_similarity.argsort()[::-1]

    recommendations = []

    for idx in indices:

        if idx == index:

            continue

        recommendations.append(idx)

        if len(recommendations) == top_n:

            break

    result = df.iloc[recommendations].copy()

    result["Similarity"] = (

        final_similarity[recommendations] * 100

    ).round(2)

    return result


# ==========================================================
# FORMAT SIMILAR PHONES
# ==========================================================

def format_similar_phones(result, phone_name):

    if result is None or result.empty:

        return "😔 Sorry, I couldn't find similar phones."

    response = f"# 📱 Phones similar to {phone_name}\n\n"

    medals = [

        "🥇",

        "🥈",

        "🥉",

        "4️⃣",

        "5️⃣"

    ]

    for i, (_, phone) in enumerate(

        result.iterrows()

    ):

        response += (

            f"## {medals[i]} {phone['Name']}\n\n"

            f"🏢 **Brand:** {phone['Company']}\n\n"

            f"💰 **Price:** ₹{phone['Price']}\n\n"

            f"⚙ **Processor:** {phone['Processor']}\n\n"

            f"🔗 **Similarity:** {phone['Similarity']}%\n\n"

        )

        response += "### Why it's similar\n\n"

        if phone["Gaming_Score"] >= 90:

            response += "✅ Excellent gaming performance\n\n"

        if phone["Camera_Score"] >= 90:

            response += "✅ Excellent camera quality\n\n"

        if phone["Performance_Score"] >= 90:

            response += "✅ Flagship-level performance\n\n"

        if phone["Battery_Score"] >= 90:

            response += "✅ Long battery life\n\n"

        response += "---\n\n"

    return response

# ==========================================================
# CHAT MEMORY
# ==========================================================

chat_memory = {

    "preferences": {

        "brand": None,
        "budget": None,
        "feature": None,
        "luxury": None

    },

    "last_phone": None,

    "last_two_phones": [],

    "last_recommendations": []

}


# ==========================================================
# UPDATE MEMORY
# ==========================================================

def update_memory(user_input):

    global chat_memory

    profile = understand_query(user_input)

    # --------------------------------------------------
    # Save preferences ONLY for recommendation queries
    # --------------------------------------------------

    if profile["intent"] == "recommend":

        if profile["brand"]:
            chat_memory["preferences"]["brand"] = profile["brand"]

        if profile["budget"]:
            chat_memory["preferences"]["budget"] = profile["budget"]

        if profile["feature"]:
            chat_memory["preferences"]["feature"] = profile["feature"]

        if profile["luxury"]:
            chat_memory["preferences"]["luxury"] = profile["luxury"]

    # --------------------------------------------------
    # Always remember phones for context
    # --------------------------------------------------

    if profile["phones"]:

        for phone in profile["phones"]:

            if phone not in chat_memory["last_two_phones"]:
                chat_memory["last_two_phones"].append(phone)

        chat_memory["last_two_phones"] = chat_memory["last_two_phones"][-2:]

        chat_memory["last_phone"] = profile["phones"][-1]

# ==========================================================
# MODIFY QUERY USING MEMORY
# ==========================================================

def modify_query(user_input):

    profile = understand_query(user_input)

    query = user_input

    # Only use remembered brand if user didn't specify one
    if profile["brand"] is None and chat_memory["preferences"]["brand"]:

        query += " " + chat_memory["preferences"]["brand"]

    # Only use remembered budget if user didn't specify one
    if profile["budget"] is None and chat_memory["preferences"]["budget"]:

        query += " under " + str(chat_memory["preferences"]["budget"])

    # Only use remembered feature if user didn't specify one
    if profile["feature"] is None and chat_memory["preferences"]["feature"]:

        query += " " + chat_memory["preferences"]["feature"]

    return query


# ==========================================================
# CONTEXT RESOLUTION
# ==========================================================

def resolve_context(user_input):

    text = user_input.lower()

    if chat_memory["last_phone"]:

        text = re.sub(

            r"\bit\b",

            chat_memory["last_phone"],

            text

        )

    if chat_memory["last_recommendations"]:

        if "first one" in text:

            text = text.replace(

                "first one",

                chat_memory["last_recommendations"][0]

            )

        if len(chat_memory["last_recommendations"]) >= 2:

            text = text.replace(

                "second one",

                chat_memory["last_recommendations"][1]

            )

        if len(chat_memory["last_recommendations"]) >= 3:

            text = text.replace(

                "third one",

                chat_memory["last_recommendations"][2]

            )

    return text

# ==========================================================
# PHONE COMPARISON
# ==========================================================

def compare_phones(phone1_name, phone2_name):

    phone1 = find_phone(phone1_name)
    phone2 = find_phone(phone2_name)

    if phone1 is None:
        return f"❌ Phone not found: {phone1_name}"

    if phone2 is None:
        return f"❌ Phone not found: {phone2_name}"

    comparison = pd.DataFrame({

        "Feature": [
            "Brand",
            "Price",
            "Processor",
            "RAM",
            "Storage",
            "Battery",
            "Charging",
            "Rear Camera",
            "Front Camera",
            "Gaming Score",
            "Camera Score",
            "Performance Score",
            "Battery Score",
            "Rating"
        ],

        phone1["Name"]: [
            str(phone1["Company"]),
            f"₹{phone1['Price']}",
            str(phone1["Processor"]),
            f"{phone1['RAM_GB']} GB",
            f"{phone1['ROM_GB']} GB",
            f"{phone1['Battery_mAh']} mAh",
            f"{phone1['Charging_Watt']} W",
            f"{phone1['Rear_Camera_MP']} MP",
            f"{phone1['Front_Camera_MP']} MP",
            str(phone1["Gaming_Score"]),
            str(phone1["Camera_Score"]),
            str(phone1["Performance_Score"]),
            str(phone1["Battery_Score"]),
            str(phone1["Star_Rating"])
        ],

        phone2["Name"]: [
            str(phone2["Company"]),
            f"₹{phone2['Price']}",
            str(phone2["Processor"]),
            f"{phone2['RAM_GB']} GB",
            f"{phone2['ROM_GB']} GB",
            f"{phone2['Battery_mAh']} mAh",
            f"{phone2['Charging_Watt']} W",
            f"{phone2['Rear_Camera_MP']} MP",
            f"{phone2['Front_Camera_MP']} MP",
            str(phone2["Gaming_Score"]),
            str(phone2["Camera_Score"]),
            str(phone2["Performance_Score"]),
            str(phone2["Battery_Score"]),
            str(phone2["Star_Rating"])
        ]
    })

    return comparison.astype(str)

# ==========================================================
# RESPONSE GENERATOR
# ==========================================================

def generate_response(user_input):

    user_input = resolve_context(user_input)

    profile = understand_query(user_input)

    print("User:", user_input)
    print("Detected phones:", profile["phones"])

    intent = profile["intent"]

    update_memory(user_input)

    # ------------------------------------------------------
    # Greeting
    # ------------------------------------------------------

    if intent == "greeting":

        return (
            "👋 Hello!\n\n"
            "I'm your AI Mobile Assistant.\n\n"
            "I can help you with:\n\n"
            "📱 Phone Recommendations\n"
            "📖 Phone Specifications\n"
            "⚖️ Phone Comparison\n"
            "🔍 Similar Phones\n\n"
            "Type **help** to see all commands."
        )

    # ------------------------------------------------------
    # Help
    # ------------------------------------------------------

    elif intent == "help":

        return (
            "# 🤖 Help\n\n"
            "Here are some things you can ask:\n\n"
            "- 🎮 Recommend a gaming phone under ₹30,000\n"
            "- 📱 Recommend Samsung phones\n"
            "- 📸 Best camera phone under ₹50,000\n"
            "- 📖 Tell me about iPhone 15\n"
            "- ⚖️ Compare iPhone 15 vs Samsung Galaxy S24\n"
            "- 🔍 Show phones similar to Google Pixel 9"
        )

    # ------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------

    elif intent == "recommend":

        query = modify_query(user_input)

        result = advanced_recommend(query)

        if result is not None and not result.empty:

            chat_memory["last_recommendations"] = (

                result["Name"].tolist()

            )

        return format_recommendations(

            result,

            understand_user(query)

        )

    # ------------------------------------------------------
    # Specification
    # ------------------------------------------------------

    elif intent == "spec":

        if len(profile["phones"]) == 0:

            return "📱 Please mention the phone name."

        return get_phone_specification(

            profile["phones"][0]

        )

    # ------------------------------------------------------
    # Comparison
    # ------------------------------------------------------

    elif intent == "compare":

        phones = profile["phones"]

        # If user says "compare both"
        if len(phones) < 2:

            phones = chat_memory["last_two_phones"]

        if len(phones) < 2:

            return "📱 Please mention two phone names."

        return compare_phones(

            phones[0],

            phones[1]

        )

    # ------------------------------------------------------
    # Similar Phones
    # ------------------------------------------------------

    elif intent == "similar":

        if len(profile["phones"]) == 0:

            return "📱 Please mention the phone name."

        result = find_similar_phones(

            profile["phones"][0]

        )

        return format_similar_phones(

            result,

            profile["phones"][0]

        )

    # ------------------------------------------------------
    # Unknown
    # ------------------------------------------------------

    return (
        "🤔 Sorry, I couldn't understand that.\n\n"
        "Try asking:\n\n"
        "- Recommend a gaming phone\n"
        "- Tell me about iPhone 15\n"
        "- Compare iPhone 15 vs Samsung Galaxy S24\n"
        "- Best camera phone under ₹50,000\n"
        "- Show phones similar to Google Pixel 9"
    )


# ==========================================================
# TERMINAL CHATBOT (OPTIONAL)
# ==========================================================

def chatbot():

    print("=" * 60)
    print("📱 AI MOBILE ASSISTANT")
    print("=" * 60)

    while True:

        user_input = input("\n👤 You: ")

        if user_input.lower() == "exit":

            print("\n👋 Goodbye!")

            break

        response = generate_response(user_input)

        print("\n🤖 Bot:\n")

        print(response)


if __name__ == "__main__":

    chatbot()