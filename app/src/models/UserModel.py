from app.src.models.DatabaseModel import DatabaseModel


class UserModel(DatabaseModel):
    username = ""
    email = ""
    firstname = ""
    lastname = ""
    id = 0
    user_type_id = 0
    status = 0
    token = ""
    gender_id = 0
    image_id = 0

    def __init__(self, id, username, firstname, lastname, email, image_id, user_type_id,
                 status, country_id, token, gender_id, date_created, date_updated, *args,
                 **kwargs):
        super().__init__("user")
        self.id = id
        self.username = username
        self.firstname = firstname
        self.lastname = lastname
        self.email = email
        self.image_id = image_id
        self.user_type_id = user_type_id
        self.status = status
        self.country_id = country_id,
        self.token = token;
        self.gender_id = gender_id;
        self.date_created = date_created
        self.date_updated = date_updated
