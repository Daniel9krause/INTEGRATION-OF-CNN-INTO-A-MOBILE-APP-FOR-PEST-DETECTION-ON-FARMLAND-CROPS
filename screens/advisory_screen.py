"""
AdvisoryScreen
---------------
A browsable reference of all 13 known pest/disease classes and their
advisory information — useful for farmers to look things up even without
scanning, e.g. while walking the field.
"""

import os
import json
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.metrics import dp
from kivy.app import App

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADVISORY_PATH = os.path.join(BASE_DIR, "data", "advisory_data.json")


class AdvisoryCard(BoxLayout):
    def __init__(self, label, info, **kwargs):
        super().__init__(orientation="vertical", size_hint_y=None, padding=dp(10),
                          spacing=dp(4), **kwargs)
        self.bind(minimum_height=self.setter("height"))

        self.add_widget(Label(text=f"[b]{label}[/b]  ({info['group']})", markup=True,
                               size_hint_y=None, height=dp(26), halign="left", valign="middle",
                               text_size=(dp(340), None)))
        self.add_widget(Label(text=info["description"], size_hint_y=None, height=dp(50),
                               font_size="12sp", halign="left", valign="top",
                               text_size=(dp(340), None)))
        self.add_widget(Label(text=f"[i]Advisory:[/i] {info['advisory']}", markup=True,
                               size_hint_y=None, height=dp(70), font_size="12sp",
                               halign="left", valign="top", text_size=(dp(340), None)))


class AdvisoryScreen(Screen):
    def on_pre_enter(self, *args):
        self.populate()

    def populate(self):
        container = self.ids.get("advisory_list")
        if not container or container.children:
            return  # only build once; content is static
        with open(ADVISORY_PATH) as f:
            advisory_db = json.load(f)
        for label, info in advisory_db.items():
            container.add_widget(AdvisoryCard(label, info))

    def go_home(self):
        App.get_running_app().root.current = "home"
