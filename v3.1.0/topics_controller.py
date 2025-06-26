import base64
from io import BytesIO
from bertopic import BERTopic
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

async def generate_topics(logs_collection, min_logs=10):
    # 1) Ensure enough logs
    count = await logs_collection.count_documents({})
    if count < min_logs:
        raise ValueError(f"Need at least {min_logs} logs to run topic modeling (found {count}).")

    # 2) Fetch messages
    records = await logs_collection.find({}, {"message": 1, "_id": 0}).to_list(length=None)
    messages = [r.get("message", "") for r in records]

    # 3) Fit BERTopic
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(messages)
    
    # 4) Compute topic frequencies using Counter instead of np.unique
    topic_counter = Counter(topics)
    topic_counts = sorted(
        topic_counter.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    #topic_info = topic_model.get_topic_info()
    #print(topic_info)
    # Build topic_info list, include Name from top words
    topic_info = []
    for topic_id, cnt in topic_counts:
        try:
            # Get topic words and join top 5
            words = topic_model.get_topic(topic_id)
            name = ", ".join([w for w, _ in words[:5]])
        except Exception:
            name = ""
        topic_info.append({
            "Topic": topic_id,
            "Count": cnt,
            "Name": name
        })

    # 5) Generate wordcloud for first valid non-outlier topic
    wordcloud_b64 = None
    for tid, cnt in topic_counts:
        if tid != -1:  # Skip outlier topic
            try:
                # Get topic words and verify type
                words = topic_model.get_topic(tid)
                if not words or not isinstance(words, list):
                    print(f"Skipping topic {tid}: invalid word list")
                    continue
                    
                # Create frequency dictionary
                freqs = {word: score for word, score in words}
                
                # Generate wordcloud
                wc = WordCloud(
                    background_color="white",
                    max_words=1000,
                    width=800,
                    height=800
                )
                wc.generate_from_frequencies(freqs)

                # Save to buffer
                buf = BytesIO()
                plt.figure(figsize=(8, 8), facecolor=None)
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                plt.tight_layout(pad=0)
                plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
                plt.close()
                buf.seek(0)
                wordcloud_b64 = base64.b64encode(buf.read()).decode()
                break  # Success, stop searching
                
            except Exception as e:
                print(f"Error generating wordcloud for topic {tid}: {e}")
                continue

    # THIS PART IS CAUSING ERRORS!!!!
    # 6) Global topic visualization HTML
    #fig = topic_model.visualize_topics()
    #viz_html = fig.to_html(full_html=False)
    
    viz_html = None

    return {
        "topic_info": topic_info,
        "top_topic_wordcloud": wordcloud_b64,
        "topics_visualization_html": viz_html
    }