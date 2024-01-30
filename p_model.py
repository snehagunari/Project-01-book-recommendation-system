# -*- coding: utf-8 -*-

import pandas as pd 
import streamlit as st

similarity_user = pd.read_pickle("C:\\Users\\sneha-pc\\Downloads\\similarity_user.pkl")
final_ratings = pd.read_excel("C:\\Users\\sneha-pc\\Downloads\\Book Recommendation\\final_ratings.xlsx")     

def recommendations_for_user(user_id):
    print('\nRecommended Books for User_id', user_id, ':\n')
    if user_id in similarity_user.columns:
        recom = list(similarity_user.sort_values(user_id, ascending=False).head().index)[1:11]
        books_list = []
        for i in recom:
            books_list += list(final_ratings[final_ratings['User-ID'] == i]['Book-Title'])
        recommended_books = set(books_list) - set(final_ratings[final_ratings['User-ID'] == user_id]['Book-Title'])
        return list(recommended_books)[:10]  # Limit the recommendations to the top 10
    else:
        return []

# Example usage
user_id = 123  # Replace with the actual user ID
recommended_books = recommendations_for_user(user_id)
print(recommended_books)  # Print recommended books

# Streamlit web app
def main():
    st.title('Book Recommendation System')
    user_id = st.number_input('Enter User ID', min_value=1, max_value=10000000, value=1)
    if st.button('Get Recommendations'):
        recommended_books = recommendations_for_user(user_id)
        st.write(f'Recommended Books for User {user_id}:')
        for book in recommended_books:
            st.write(book)

if __name__ == '__main__':
    main()