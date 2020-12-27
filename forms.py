from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired


class SearchForm(FlaskForm):
    search_box = StringField('What film are you looking for? (only original name)', validators=[DataRequired()])
    submit = SubmitField('Search')
