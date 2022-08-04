"""empty message

Revision ID: 4a344d1a9965
Revises: 855824996112
Create Date: 2022-07-01 11:59:56.441234

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4a344d1a9965'
down_revision = '855824996112'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('movies', sa.Column('movie_desc', sa.String(length=1000), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('movies', 'movie_desc')
    # ### end Alembic commands ###