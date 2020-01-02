import unittest
import time
import sys
sys.path.append('/root/ML-self-model')
sys.path.append('/root/ML-self-model/app')
from app import create_app, db
from app.models import User, AnonymousUser, Role, Permission
from app.fake import users,posts

class UserModelTestCase(unittest.TestCase):
    def setUp(self):
        # self.app = create_app('production')
        self.app = create_app('development')
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        self.app_context.pop()

    def test_password_setter(self):
        pass

    def test_no_password_getter(self):
        print('OK!')
        Role.insert_roles()