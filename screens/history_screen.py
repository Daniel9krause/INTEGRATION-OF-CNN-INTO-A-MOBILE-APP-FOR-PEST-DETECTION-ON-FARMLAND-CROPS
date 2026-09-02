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

from utils.result_presentation import present


def _row_value(row, key, default=None):
    """sqlite3.Row has no .get(), and history written by the previous build
    predates the status/headline/detail columns."""
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    return default if value is None else value


class HistoryRow(BoxLayout):
    def __init__(self, scan_row, on_select, **kwargs):
        super().__init__(orientation="horizontal", size_hint_y=None, height=dp(80),
                          padding=dp(6), spacing=dp(10), **kwargs)
        self.scan_row = scan_row

        thumb = AsyncImage(source=scan_row["image_path"], size_hint_x=0.25, allow_stretch=True)
        self.add_widget(thumb)

        # Refused scans show no percentage. A row reading "Beetle • 62.8%"
        # for a photo the app actually refused to diagnose is the same
        # false reassurance the result screen was fixed to avoid, just one
        # screen further away from where it was explained.
        status = _row_value(scan_row, "status", "ok")
        view = present(status, scan_row["predicted_label"], scan_row["confidence"],
                       _row_value(scan_row, "headline", ""),
                       _row_value(scan_row, "detail", ""))
        if view["show_confidence"]:
            subtitle = (f"{view['group_text']} • {scan_row['confidence'] * 100:.1f}%"
                        f"  |  {scan_row['created_at']}")
        else:
            subtitle = f"{view['group_text']}  |  {scan_row['created_at']}"

        info = BoxLayout(orientation="vertical", size_hint_x=0.55)
        info.add_widget(Label(text=f"[b]{view['display_label']}[/b]", markup=True,
                               color=view["confidence_color"],
                               halign="left", valign="middle", size_hint_y=0.5))
        info.add_widget(Label(text=subtitle, font_size="11sp",
                               halign="left", valign="middle", size_hint_y=0.5))
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
        # Re-render a saved result without re-running inference, through the
        # same presentation logic a fresh scan uses. The stored `status` is
        # what drives it: a scan that was refused at capture time ("No plant
        # detected", "Too dark") redisplays as a refusal here too. Deriving
        # this from the confidence number - as this screen used to - could
        # resurrect a refused scan as a confident diagnosis, since the row is
        # stored under a real class name either way.
        label = scan_row["predicted_label"]
        confidence = scan_row["confidence"]
        status = _row_value(scan_row, "status", "ok")
        view = present(status, label, confidence,
                       _row_value(scan_row, "headline", ""),
                       _row_value(scan_row, "detail", ""))

        result_screen.image_path = scan_row["image_path"]
        result_screen.label_text = view["display_label"]
        result_screen.confidence_color = view["confidence_color"]
        result_screen.group_text = view["group_text"]
        result_screen.description_text = view["description_text"]
        result_screen.advisory_text = view["advisory_text"]
        result_screen.show_confidence = view["show_confidence"]
        result_screen.show_advisory = view["show_advisory"]
        result_screen.show_flag_button = view["show_flag_button"]
        if view["show_confidence"]:
            result_screen.confidence_text = f"{confidence * 100:.1f}% confidence"
            result_screen.confidence_value = confidence * 100
        else:
            result_screen.confidence_text = ""
            result_screen.confidence_value = 0

        app.root.current = "result"

    def clear_history(self):
        App.get_running_app().db.clear_history()
        self.refresh_list()

    def go_home(self):
        App.get_running_app().root.current = "home"
