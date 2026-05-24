from src.auth.users import DUMMY_USERS
from src.auth.session_store import authenticated_users

def login_user(email, password):

    if email not in DUMMY_USERS:
        return False

    if DUMMY_USERS[email] != password:
        return False

    authenticated_users.add(email)

    return True


def is_authenticated(email):

    return email in authenticated_users