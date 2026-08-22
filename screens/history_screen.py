"""
HistoryScreen
--------------
Lists every past scan pulled from SQLite, newest first: thumbnail,
predicted label, group, confidence, and timestamp. Tapping a row reopens
that scan's full result + advisory.
"""

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import AsyncImage
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.app import App
from kivy.metrics import dp

from screens.result_screen import build_result_info


class HistoryRow(BoxLayout):
    def __init__(self, scan_row, on_select, **kwargs):
        super().__init__(orientation="horizontal", size_hint_y=None, height=dp(80),
                          padding=dp(6), spacing=dp(10), **kwargs)
        self.scan_row = scan_row

        thumb = AsyncImage(source=scan_row["image_path"], size_hint_x=0.25, allow_stretch=True)
        self.add_widget(thumb)

        info = BoxLayout(orientation="vertical", size_hint_x=0.55)
        info.add_widget(Label(text=f"[b]{scan_row['predicted_label']}[/b]", markup=True,
                               halign="left", valign="middle", size_hint_y=0.5))
        info.add_widget(Label(text=f"{scan_row['group_name']} • {scan_row['confidence']*100:.1f}%  |  {scan_row['created_at']}",
                               font_size="11sp", halign="left", valign="middle", size_hint_y=0.5))
        self.add_widget(info)

        view_btn = Button(text="View", size_hint_x=0.2)
        view_btn.bind(on_release=lambda *_: on_select(scan_row))
        self.add_widget(view_btn)


class HistoryScreen(Screen):
    def on_pre_enter(self, *args):
        self.refresh_list()

    def refresh_list(self):
        container = self.ids.get("history_list")
        if not container:
            return
        container.clear_widgets()

        app = App.get_running_app()
        scans = app.db.get_all_scans()

        if not scans:
            container.add_widget(Label(text="No scans yet — capture your first image!",
                                        size_hint_y=None, height=dp(60)))
            return

        for scan in scans:
            container.add_widget(HistoryRow(scan, self._open_scan))

    def _open_scan(self, scan_row):
        app = App.get_running_app()
        result_screen = app.root.get_screen("result")
        # Re-render a saved result without re-running inference. Reuses the
        # same confidence-threshold logic ResultScreen uses on a fresh scan,
        # so a low-confidence result still shows as "Uncertain" here too,
        # not as a confident match just because it's stored under a real
        # class name.
        label = scan_row["predicted_label"]
        confidence = scan_row["confidence"]
        display_label, info, color = build_result_info(label, confidence)

        result_screen.image_path = scan_row["image_path"]
        result_screen.label_text = display_label
        result_screen.confidence_text = f"{confidence * 100:.1f}% confidence"
        result_screen.confidence_value = confidence * 100
        result_screen.confidence_color = color
        result_screen.group_text = info["group"]
        result_screen.description_text = info["description"]
        result_screen.advisory_text = info["advisory"]

        app.root.current = "result"

    def clear_history(self):
        App.get_running_app().db.clear_history()
        self.refresh_list()

    def go_home(self):
        App.get_running_app().root.current = "home"
