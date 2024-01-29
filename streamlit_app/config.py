"""

Config file for Streamlit App

"""

from streamlit_app.member import Member


TITLE = "Projet : Reconnaissance de plantes"

TEAM_MEMBERS = [
    Member(
        name="Julie Le Vu",
        linkedin_url="https://www.linkedin.com/in/julie-le-vu-201b2a261/",
    ),
    Member(
        name="Hadrien Gremillet",
        linkedin_url="https://www.linkedin.com/in/hadrieng/",
    ),
    Member(
        name="Florent Maurice",
        linkedin_url="https://www.linkedin.com/in/florent-maurice-086a93274/",
    ),
]

PROMOTION = "Promotion Bootcamp Data Scientist - Mai 2023"
