"""
HistoryScreen
--------------
Lists every past scan pulled from SQLite, newest first: thumbnail,
predicted label, group, confidence, and timestamp. Tapping View reopens
that scan's full result + advisory; Remove deletes that one scan.

Removal is per-row rather than all-or-nothing because history is a working
list, not an archive: a farmer photographs the same plant three times to get
one usable shot, and wants the two blurred attempts gone without losing the
month of scans underneath them.
"""

import os

from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import AsyncImage
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.app import App
from kivy.metrics import dp

from screens.home_screen import CAPTURE_DIR
from utils.result_presentation import present


def _row_value(row, key, default=None):
    """sqlite3.Row has no .get(), and history written by the previous build
    predates the status/headline/detail columns."""
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    return default if value is None else value


def _view_of(scan_row):
    """How this scan was presented to the farmer at the time - the same call
    the row and the result screen make, so a refused scan is never named
    after a pest anywhere, including in a "Remove this?" prompt."""
    return present(_row_value(scan_row, "status", "ok"),
                   scan_row["predicted_label"], scan_row["confidence"],
                   _row_value(scan_row, "headline", ""),
                   _row_value(scan_row, "detail", ""))


def _discard_capture(image_path):
    """Delete the photo behind a removed scan, but only if it is one of ours.

    Every capture, and every picture picked through Upload, is copied into
    CAPTURE_DIR before it is ever recorded (see HomeScreen._copy_android_uri),
    so a path outside that folder is one this app did not create and has no
    business unlinking - deleting a farmer's original gallery photo because
    they tidied up a scan list would be unforgivable.

    The training copy under collected_data/ is a separate file - DataCollector
    copies rather than moves - so removing a row from history does not throw
    away the dataset it contributed to.
    """
    if not image_path:
        return
    try:
        captures = os.path.realpath(CAPTURE_DIR)
        target = os.path.realpath(image_path)
        # commonpath raises ValueError when the two sit on different Windows
        # drives, which is itself a "not ours" answer.
        if os.path.commonpath([captures, target]) != captures:
            return
        if os.path.exists(target):
            os.remove(target)
    except (OSError, ValueError) as e:
        # File already gone, SD card pulled, permission revoked. The row -
        # which is what the farmer actually asked to be rid of - is already
        # deleted, so this is worth logging and nothing more.
        print(f"[HistoryScreen] Could not delete {image_path}: {e}")


class HistoryRow(BoxLayout):
    def __init__(self, scan_row, on_select, on_delete, **kwargs):
        super().__init__(orientation="horizontal", size_hint_y=None, height=dp(80),
                          padding=dp(6), spacing=dp(8), **kwargs)
        self.scan_row = scan_row

        thumb = AsyncImage(source=scan_row["image_path"], size_hint_x=0.22, allow_stretch=True)
        self.add_widget(thumb)

        # Refused scans show no percentage. A row reading "Beetle - 62.8%"
        # for a photo the app actually refused to diagnose is the same
        # false reassurance the result screen was fixed to avoid, just one
        # screen further away from where it was explained.
        view = _view_of(scan_row)
        if view["show_confidence"]:
            subtitle = (f"{view['group_text']} • {scan_row['confidence'] * 100:.1f}%"
                        f"  |  {scan_row['created_at']}")
        else:
            subtitle = f"{view['group_text']}  |  {scan_row['created_at']}"

        info = BoxLayout(orientation="vertical", size_hint_x=0.44)
        info.add_widget(Label(text=f"[b]{view['display_label']}[/b]", markup=True,
                               color=view["confidence_color"],
                               halign="left", valign="middle", size_hint_y=0.5))
        info.add_widget(Label(text=subtitle, font_size="11sp",
                               halign="left", valign="middle", size_hint_y=0.5))
        self.add_widget(info)

        view_btn = Button(text="View", font_size="13sp", size_hint_x=0.17)
        view_btn.bind(on_release=lambda *_: on_select(scan_row))
        self.add_widget(view_btn)

        # Coloured apart from View, because on a phone the two sit a thumb's
        # width from each other and only one of them can be undone.
        remove_btn = Button(text="Remove", font_size="13sp", size_hint_x=0.17,
                             background_color=(0.72, 0.24, 0.20, 1))
        remove_btn.bind(on_release=lambda *_: on_delete(scan_row))
        self.add_widget(remove_btn)


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
            container.add_widget(HistoryRow(scan, self._open_scan, self._confirm_delete))

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
        confidence = scan_row["confidence"]
        view = _view_of(scan_row)

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

    def _confirm(self, title, message, confirm_text, on_confirm):
        """A yes/no dialog for the two destructive buttons on this screen.

        Both deletions are permanent, and Remove sits a few millimetres from
        View on a phone - a scan taken out in the field cannot be retaken
        from the house, so a mis-tap must not be enough to lose it.
        """
        body = BoxLayout(orientation="vertical", padding=dp(14), spacing=dp(14))
        body.add_widget(Label(text=message, markup=True, halign="center",
                               valign="middle", text_size=(dp(230), None)))

        popup = Popup(title=title, content=body, size_hint=(0.88, None),
                      height=dp(220))

        buttons = BoxLayout(size_hint_y=None, height=dp(46), spacing=dp(10))
        cancel_btn = Button(text="Cancel")
        cancel_btn.bind(on_release=lambda *_: popup.dismiss())
        confirm_btn = Button(text=confirm_text, background_color=(0.72, 0.24, 0.20, 1))

        def _do(*_):
            popup.dismiss()
            on_confirm()

        confirm_btn.bind(on_release=_do)
        buttons.add_widget(cancel_btn)
        buttons.add_widget(confirm_btn)
        body.add_widget(buttons)
        popup.open()

    def _confirm_delete(self, scan_row):
        view = _view_of(scan_row)
        self._confirm(
            "Remove scan",
            f"Remove this scan?\n\n[b]{view['display_label']}[/b]\n{scan_row['created_at']}",
            "Remove",
            lambda: self._delete_scan(scan_row),
        )

    def _delete_scan(self, scan_row):
        app = App.get_running_app()
        app.db.delete_scan(scan_row["id"])
        _discard_capture(scan_row["image_path"])
        self.refresh_list()

    def clear_history(self):
        self._confirm(
            "Clear history",
            "Remove [b]all[/b] saved scans?\n\nThis cannot be undone.",
            "Clear all",
            self._clear_history,
        )

    def _clear_history(self):
        app = App.get_running_app()
        # Read the rows before dropping them: clearing the list used to leave
        # every photo behind on the phone, so a farmer who "cleared" their
        # history still had months of captures eating storage - and, on a
        # shared phone, still sitting on disk.
        paths = [row["image_path"] for row in app.db.get_all_scans(limit=1000000)]
        app.db.clear_history()
        for path in paths:
            _discard_capture(path)
        self.refresh_list()

    def go_home(self):
        App.get_running_app().root.current = "home"
