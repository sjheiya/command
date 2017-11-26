import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.Qsci import QsciScintilla, QsciLexerPython,QsciLexerCustom
from PyQt5.Qt import QApplication
import keyword
import builtins
import re
import functions
import data
import time
# Try importing the Cython module
try:
    import cython_lexers
    cython_found = True
except Exception as ex:
    print(ex)
    cython_found = False
# Try importing the Nim module
try:
    import nim_lexers
    nim_found = True
except Exception as ex:
    print(ex)
    nim_found = False

def set_font(lexer, style_name, style_options):
    font, color, size, bold = style_options
    lexer.setColor(
        QColor(color),
        lexer.styles[style_name]
    )
    weight = QFont.Normal
    if bold == 1 or bold == True:
        weight = QFont.Bold
    elif bold == 2:
        weight = QFont.Black
    lexer.setFont(
        QFont(font, size, weight=weight),
        lexer.styles[style_name]
    )

class CustomPython(QsciLexerCustom):
    class Sequence:
        def __init__(self,
                     start,
                     stop_sequences,
                     stop_characters,
                     style,
                     add_to_style):
            self.start = start
            self.stop_sequences = stop_sequences
            self.stop_characters = stop_characters
            self.style = style
            self.add_to_style = add_to_style

    # Class variables
    # Lexer index counter for Nim styling
    _index = 0
    index = 0
    # Styles
    styles = {
        "Default": 0,
        "Comment": 1,
        "Number": 2,
        "DoubleQuotedString": 3,
        "SingleQuotedString": 4,
        "Keyword": 5,
        "TripleSingleQuotedString": 6,
        "TripleDoubleQuotedString": 7,
        "ClassName": 8,
        "FunctionMethodName": 9,
        "Operator": 10,
        "Identifier": 11,
        "CommentBlock": 12,
        "UnclosedString": 13,
        "HighlightedIdentifier": 14,
        "Decorator": 15,
        "CustomKeyword": 16,
    }
    default_color = QColor(0,0,0)
    default_paper = QColor(255,255,255)
    default_font = QFont('Courier', 10)
    # Styling lists and characters
    keyword_list = list(set(keyword.kwlist + dir(builtins)))
    additional_list = []
    sq = Sequence('\'', ['\'', '\n'], [], styles["SingleQuotedString"], True)
    dq = Sequence('"', ['"', '\n'], [], styles["DoubleQuotedString"], True)
    edq = Sequence('""', [], [], styles["DoubleQuotedString"], True)
    esq = Sequence('\'\'', [], [], styles["DoubleQuotedString"], True)
    tqd = Sequence('\'\'\'', ['\'\'\''], [], styles["TripleSingleQuotedString"], True)
    tqs = Sequence('"""', ['"""'], [], styles["TripleDoubleQuotedString"], True)
    cls = Sequence('class', [':'], ['(', '\n'], styles["ClassName"], False)
    defi = Sequence('def', [], ['('], styles["FunctionMethodName"], False)
    comment = Sequence('#', [], ['\n'], styles["Comment"], True)
    dcomment = Sequence('##', [], ['\n'], styles["CommentBlock"], True)
    decorator = Sequence('@', ['\n'], [' '], styles["Decorator"], True)
    sequence_lists = [
        sq, dq, edq, esq, tqd, tqs, cls, defi, comment, dcomment, decorator
    ]
    multiline_sequence_list = [tqd, tqs]
    sequence_start_chrs = [x.start for x in sequence_lists]
    # Regular expression split sequence to tokenize text
    splitter = re.compile(r"(\\'|\\\"|\(\*|\*\)|\n|\"+|\'+|\#+|\s+|\w+|\W)")
    # Characters that autoindent one level on pressing Return/Enter
    autoindent_characters = [":"]

    def __init__(self, parent=None, additional_keywords=[]):
        """Overridden initialization"""
        # Initialize superclass
        super().__init__()
        # Set the lexer's index
        self.index = CustomPython._index
        CustomPython._index += 1
        # Set the additional keywords
        self.additional_list = ["self"]
        self.additional_list.extend(additional_keywords)
        if nim_found == True:
            nim_lexers.python_set_keywords(self.index, additional_keywords)
        # Set the default style values
        self.setDefaultColor(self.default_color)
        self.setDefaultPaper(self.default_paper)
        self.setDefaultFont(self.default_font)
        # Reset autoindentation style
        self.setAutoIndentStyle(0)
        # Set the theme
        self.set_theme(data.theme)

    def language(self):
        return "Python"

    def description(self, style):
        if style <= 16:
            description = "Custom lexer for the Python programming languages"
        else:
            description = ""
        return description

    def set_theme(self, theme):
        for style in self.styles:
            # Papers
            paper = QColor(getattr(theme.Paper.Python, style))
            self.setPaper(paper, self.styles[style])
            # Fonts
            set_font(self, style, getattr(theme.Font.Python, style))

    if nim_found == True:
        def __del__(self):
            nim_lexers.python_delete_keywords(self.index)

        def styleText(self, start, end):
            nim_lexers.python_style_text(self.index, start, end, self, self.editor())
    else:
        def styleText(self, start, end):
            editor = self.editor()
            if editor is None:
                return
            # Initialize the styling
            self.startStyling(start)
            # Scintilla works with bytes, so we have to adjust the start and end boundaries
            text = bytearray(editor.text(), "utf-8")[start:end].decode("utf-8")
            # Loop optimizations
            setStyling = self.setStyling
            # Initialize comment state and split the text into tokens
            sequence = None
            tokens = [(token, len(bytearray(token, "utf-8"))) for token in self.splitter.findall(text)]
            # Check if there is a style(comment, string, ...) stretching on from the previous line
            if start != 0:
                previous_style = editor.SendScintilla(editor.SCI_GETSTYLEAT, start - 1)
                for i in self.multiline_sequence_list:
                    if previous_style == i.style:
                        sequence = i
                        break

            # Style the tokens accordingly
            for i, token in enumerate(tokens):
                #                print(token[0].encode("utf-8"))
                token_name = token[0]
                token_length = token[1]
                if sequence != None:
                    if token_name in sequence.stop_sequences:
                        if sequence.add_to_style == True:
                            setStyling(token_length, sequence.style)
                        else:
                            setStyling(token_length, self.styles["Default"])
                        sequence = None
                    elif any(ch in token_name for ch in sequence.stop_characters):
                        if sequence.add_to_style == True:
                            setStyling(token_length, sequence.style)
                        else:
                            setStyling(token_length, self.styles["Default"])
                        sequence = None
                    else:
                        setStyling(token_length, sequence.style)
                elif token_name in self.sequence_start_chrs:
                    for i in self.sequence_lists:
                        if token_name == i.start:
                            if i.stop_sequences == [] and i.stop_characters == []:
                                # Skip styling if both stop sequences and stop characters are empty
                                setStyling(token_length, i.style)
                            else:
                                # Style the sequence and store the reference to it
                                sequence = i
                                if i.add_to_style == True:
                                    setStyling(token_length, sequence.style)
                                else:
                                    if token_name in self.keyword_list:
                                        setStyling(token_length, self.styles["Keyword"])
                                    elif token_name in self.additional_list:
                                        setStyling(token_length, self.styles["CustomKeyword"])
                                    else:
                                        setStyling(token_length, self.styles["Default"])
                            break
                elif token_name in self.keyword_list:
                    setStyling(token_length, self.styles["Keyword"])
                elif token_name in self.additional_list:
                    setStyling(token_length, self.styles["CustomKeyword"])
                elif token_name[0].isdigit():
                    setStyling(token_length, self.styles["Number"])
                else:
                    setStyling(token_length, self.styles["Default"])


class SimplePythonEditor(QsciScintilla):
    ARROW_MARKER_NUM = 8

    def __init__(self, parent=None):
        super(SimplePythonEditor, self).__init__(parent)

        # Set the default font
        font = QFont()
        font.setFamily('Courier')
        font.setFixedPitch(True)
        font.setPointSize(10)
        self.setFont(font)
        self.setMarginsFont(font)

        # Margin 0 is used for line numbers
        fontmetrics = QFontMetrics(font)
        self.setMarginsFont(font)
        self.setMarginWidth(0, fontmetrics.width("00000") + 6)
        self.setMarginLineNumbers(0, True)
        self.setMarginsBackgroundColor(QColor("#cccccc"))

        # Clickable margin 1 for showing markers
        self.setMarginSensitivity(1, True)
        # self.connect(self,
        #    SIGNAL('marginClicked(int, int, Qt::KeyboardModifiers)'),
        #    self.on_margin_clicked)
        self.marginClicked.connect(self.on_margin_clicked)
        self.markerDefine(QsciScintilla.RightArrow,
                          self.ARROW_MARKER_NUM)
        self.setMarkerBackgroundColor(QColor("#ee1111"),
                                      self.ARROW_MARKER_NUM)

        # Brace matching: enable for a brace immediately before or after
        # the current position
        #
        self.setBraceMatching(QsciScintilla.SloppyBraceMatch)

        # Current line visible with special background color
        self.setCaretLineVisible(True)
        self.setCaretLineBackgroundColor(QColor("#ffe4e4"))

        # Set Python lexer
        # Set style for Python comments (style number 1) to a fixed-width
        # courier.
        #
        lexer = CustomPython()
        lexer.setDefaultFont(font)
        self.setLexer(lexer)
        self.SendScintilla(QsciScintilla.SCI_STYLESETFONT, 1, bytes('Courier', "utf-8"))
        self.SendScintilla(QsciScintilla.SCI_SETPROPERTY, bytes("fold", "utf-8"), bytes("1", "utf-8"))
        self.setFolding(QsciScintilla.BoxedTreeFoldStyle)

        # Don't want to see the horizontal scrollbar at all
        # Use raw message to Scintilla here (all messages are documented
        # here: http://www.scintilla.org/ScintillaDoc.html)
        self.SendScintilla(QsciScintilla.SCI_SETHSCROLLBAR, 0)

        # not too small
        self.setMinimumSize(600, 450)

    def on_margin_clicked(self, nmargin, nline, modifiers):

        # Toggle marker for the line the margin was clicked on
        if self.markersAtLine(nline) != 0:
            self.markerDelete(nline, self.ARROW_MARKER_NUM)
        else:
            self.markerAdd(nline, self.ARROW_MARKER_NUM)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = SimplePythonEditor()
    editor.show()
    editor.setText(open(sys.argv[0]).read())
    app.exec_()