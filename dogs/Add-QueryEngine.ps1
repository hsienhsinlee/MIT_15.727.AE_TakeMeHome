<#
.SYNOPSIS
    Embeds an interactive Dog LOS Query Engine into the Take Me Home report.

.DESCRIPTION
    - Opens the .docx in Word (silently)
    - Appends a new page with a fully interactive form:
        * Dropdown content controls for Sex, Intake Condition, Intake Type
        * Text content controls for Breed, Color, Age
        * A clickable "Run Query" MACROBUTTON
        * A live Results box that updates in-document
    - Embeds the PredictDogLOS VBA macro directly in the file
    - Saves as .docm (macro-enabled)

.NOTES
    Run from Windows PowerShell (NOT WSL).
    Requires: Microsoft Word installed on this machine.
    One-time setup: In Word > File > Options > Trust Center > Trust Center Settings
                    > Macro Settings > check "Trust access to the VBA project object model"
#>

# ── Paths ─────────────────────────────────────────────────────────────────────
$InputPath  = "\\wsl.localhost\Ubuntu\home\leehs\GitHub\MIT_Analytical_Edge_Sp26\project\Claude work\final report\TakeMeHome_FinalReport vba.BACKUP.docx"
$OutputPath = "\\wsl.localhost\Ubuntu\home\leehs\GitHub\MIT_Analytical_Edge_Sp26\project\Claude work\final report\TakeMeHome_FinalReport vba.docm"

# ── VBA macro code ────────────────────────────────────────────────────────────
$VBACode = @'
' ============================================================================
'  PredictDogLOS  --  Take Me Home Dog Adoption LOS Predictor
'  Reads form values from Content Controls in the document, POSTs to the
'  Flask API at http://127.0.0.1:5000/predict, and writes the result back
'  into the document's "QueryResult" bookmark.
'  Pre-requisite: python app.py must be running in a terminal.
' ============================================================================

Sub PredictDogLOS()

    Dim doc         As Document
    Dim http        As Object
    Dim jsonBody    As String
    Dim response    As String

    ' -- Read content controls by Tag ----------------------------------------
    Dim breed       As String : breed      = GetCC("breed")
    Dim color       As String : color      = GetCC("color")
    Dim ageDays     As String : ageDays    = GetCC("age_days")
    Dim sex         As String : sex        = GetCC("sex")
    Dim intakeCond  As String : intakeCond = GetCC("intake_condition")
    Dim intakeType  As String : intakeType = GetCC("intake_type")

    If breed = "" Or ageDays = "" Then
        MsgBox "Please fill in at least Breed and Age before running the query.", _
               vbExclamation, "Take Me Home"
        Exit Sub
    End If

    ' -- Build JSON body -------------------------------------------------------
    jsonBody = "{" & _
        """breed_grouped"":"    & Chr(34) & breed      & Chr(34) & "," & _
        """primary_color"":"    & Chr(34) & color      & Chr(34) & "," & _
        """age_days"":"         & ageDays              & ","           & _
        """sex_upon_intake"":"  & Chr(34) & sex        & Chr(34) & "," & _
        """Intake Condition"":""& Chr(34) & intakeCond & Chr(34) & "," & _
        """Intake Type"":"      & Chr(34) & intakeType & Chr(34) & _
        "}"

    ' -- POST to Flask ---------------------------------------------------------
    Set http = CreateObject("MSXML2.XMLHTTP")
    http.Open "POST", "http://127.0.0.1:5000/predict", False
    http.setRequestHeader "Content-Type", "application/json"

    On Error GoTo ApiError
    http.Send jsonBody

    ' -- Parse response --------------------------------------------------------
    response = http.responseText

    Dim predDays    As String : predDays  = ExtractValue(response, "predicted_days")
    Dim predWeeks   As String : predWeeks = ExtractValue(response, "predicted_weeks")
    Dim rangeLow    As String : rangeLow  = ExtractValue(response, "range_low_days")
    Dim rangeHigh   As String : rangeHigh = ExtractValue(response, "range_high_days")
    Dim mae         As String : mae       = ExtractValue(response, "mae_days")

    ' -- Write result into the document ----------------------------------------
    Dim resultText As String
    resultText = "PREDICTION RESULT" & vbCrLf & _
                 String(40, "-") & vbCrLf & _
                 "Breed:       " & breed      & vbCrLf & _
                 "Age:         " & ageDays    & " days" & vbCrLf & _
                 "Sex:         " & sex        & vbCrLf & _
                 "Condition:   " & intakeCond & vbCrLf & _
                 "Intake:      " & intakeType & vbCrLf & _
                 String(40, "-") & vbCrLf & _
                 "Predicted:   " & predDays & " days  (" & predWeeks & " weeks)" & vbCrLf & _
                 "Range:       " & rangeLow & "  to  " & rangeHigh & " days" & vbCrLf & _
                 "Model MAE:   +/- " & mae & " days"

    ' Update the QueryResult content control
    Dim cc As ContentControl
    For Each cc In ActiveDocument.ContentControls
        If cc.Tag = "query_result" Then
            cc.LockContents = False
            cc.Range.Text = resultText
            cc.LockContents = True
        End If
    Next cc

    Set http = Nothing
    Exit Sub

ApiError:
    MsgBox "Could not reach the Flask server at http://127.0.0.1:5000" & vbCrLf & vbCrLf & _
           "Make sure you have started the model server:" & vbCrLf & _
           "  Open a terminal and run:  python app.py", _
           vbCritical, "Connection Error"
    Set http = Nothing
End Sub


' -- Helper: read value from a Content Control by Tag -------------------------
Private Function GetCC(tag As String) As String
    Dim cc As ContentControl
    For Each cc In ActiveDocument.ContentControls
        If cc.Tag = tag Then
            GetCC = Trim(cc.Range.Text)
            Exit Function
        End If
    Next cc
    GetCC = ""
End Function


' -- Helper: extract a numeric value from a flat JSON string ------------------
Private Function ExtractValue(json As String, key As String) As String
    Dim pos    As Long
    Dim start  As Long
    Dim finish As Long
    pos = InStr(json, Chr(34) & key & Chr(34))
    If pos = 0 Then ExtractValue = "N/A" : Exit Function
    start = InStr(pos + Len(key) + 2, json, ":") + 1
    Do While Mid(json, start, 1) = " " : start = start + 1 : Loop
    finish = start
    Do While finish <= Len(json) _
         And Mid(json, finish, 1) <> "," _
         And Mid(json, finish, 1) <> "}"
        finish = finish + 1
    Loop
    ExtractValue = Trim(Mid(json, start, finish - start))
End Function
'@

# ── Constants (Word enum values) ──────────────────────────────────────────────
$wdStory                     = 6
$wdPageBreak                 = 7
$wdCollapseEnd               = 0
$wdContentControlText        = 1
$wdContentControlDropdownList= 3
$wdContentControlRichText    = 0
$wdFormatDocumentMacroEnabled= 13
$wdAlignParagraphCenter      = 1
$wdAlignParagraphLeft        = 0

Write-Host ""
Write-Host "Take Me Home -- Query Engine Installer"
Write-Host "======================================="
Write-Host "Input:  $InputPath"
Write-Host "Output: $OutputPath"
Write-Host ""

# ── Launch Word ───────────────────────────────────────────────────────────────
Write-Host "Launching Word..."
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0  # wdAlertsNone

try {

    # ── Open document ─────────────────────────────────────────────────────────
    Write-Host "Opening document..."
    $doc = $word.Documents.Open($InputPath)

    # ── Remove any previously added Section 5 (idempotent) ───────────────────
    # (skip if this is the first run from the backup)

    # ── Embed VBA macro ───────────────────────────────────────────────────────
    Write-Host "Embedding VBA macro..."
    try {
        $vbaComponent = $doc.VBProject.VBComponents.Add(1)  # vbext_ct_StdModule
        $vbaComponent.Name = "TakeMeHome"
        $vbaComponent.CodeModule.AddFromString($VBACode)
        Write-Host "  VBA module 'TakeMeHome' added."
    } catch {
        Write-Warning "Could not add VBA module. Enable 'Trust access to the VBA project object model':"
        Write-Warning "  Word > File > Options > Trust Center > Trust Center Settings > Macro Settings"
        throw
    }

    # ── Move to end of document ───────────────────────────────────────────────
    $sel = $word.Selection
    $sel.EndKey($wdStory) | Out-Null

    # ── Page break ────────────────────────────────────────────────────────────
    $sel.InsertBreak($wdPageBreak)

    # ── Section heading ───────────────────────────────────────────────────────
    $sel.Style = $doc.Styles("Heading 1")
    $sel.TypeText("5.  Interactive Dog LOS Query Engine")
    $sel.TypeParagraph()

    $sel.Style = $doc.Styles("Normal")
    $sel.TypeText("Fill in the dog attributes below and click Run Query. The model server (python app.py) must be running in a terminal.")
    $sel.TypeParagraph()
    $sel.TypeParagraph()

    # ── How-to note ───────────────────────────────────────────────────────────
    $sel.Style = $doc.Styles("Heading 2")
    $sel.TypeText("5.1  Input Form")
    $sel.TypeParagraph()
    $sel.Style = $doc.Styles("Normal")

    # ── Form table (6 input rows + 1 header) ──────────────────────────────────
    Write-Host "Building input form table..."
    $table = $doc.Tables.Add($sel.Range, 7, 2)
    $table.Borders.Enable = $true
    $table.PreferredWidthType = 2   # wdPreferredWidthPercent
    $table.PreferredWidth = 80

    # Column widths
    $table.Columns(1).PreferredWidthType = 2
    $table.Columns(1).PreferredWidth = 30
    $table.Columns(2).PreferredWidthType = 2
    $table.Columns(2).PreferredWidth = 70

    # Header row
    $headerCell1 = $table.Cell(1,1)
    $headerCell2 = $table.Cell(1,2)
    $headerCell1.Range.Text = "Attribute"
    $headerCell2.Range.Text = "Your Input"
    $headerCell1.Range.Bold = $true
    $headerCell2.Range.Bold = $true
    # Shade header (dark blue)
    $headerCell1.Shading.BackgroundPatternColor = 0x572E2E  # dark blue (BGR)
    $headerCell2.Shading.BackgroundPatternColor = 0x572E2E
    $headerCell1.Range.Font.Color = 16777215  # white
    $headerCell2.Range.Font.Color = 16777215

    # Helper to add a label in column 1
    function Set-Label($row, $label, $hint) {
        $cell = $table.Cell($row, 1)
        $cell.Range.Text = $label
        $cell.Range.Bold = $true
        if ($hint) {
            $cell.Range.Text = "$label`r`n"
            $run2 = $cell.Range
            $run2.Collapse($wdCollapseEnd)
            $run2.Text = $hint
            $run2.Font.Size = 8
            $run2.Font.Italic = $true
            $run2.Font.Bold = $false
            $run2.Font.Color = 6710886  # grey
        }
    }

    # Helper to add a text content control in column 2
    function Add-TextCC($row, $tag, $placeholder, $defaultVal) {
        $cell  = $table.Cell($row, 2)
        $range = $cell.Range
        $range.Text = ""
        $range.Collapse($wdCollapseEnd)
        $cc = $doc.ContentControls.Add($wdContentControlText, $range)
        $cc.Tag   = $tag
        $cc.Title = $tag
        $cc.PlaceholderText.SetPlaceholderText("", $null, $placeholder)
        # Set default value
        $cc.Range.Text = $defaultVal
        return $cc
    }

    # Helper to add a dropdown content control in column 2
    function Add-DropdownCC($row, $tag, [string[]]$items, $defaultItem) {
        $cell  = $table.Cell($row, 2)
        $range = $cell.Range
        $range.Text = ""
        $range.Collapse($wdCollapseEnd)
        $cc = $doc.ContentControls.Add($wdContentControlDropdownList, $range)
        $cc.Tag   = $tag
        $cc.Title = $tag
        $i = 1
        foreach ($item in $items) {
            $cc.DropdownListEntries.Add($item, $item, $i) | Out-Null
            $i++
        }
        # Set default to first real item
        $cc.DropdownListEntries.Item(1).Select()
        return $cc
    }

    # Row 2: Breed
    Set-Label 2 "Breed" "(full breed name, lowercase)"
    Add-TextCC 2 "breed" "e.g. labrador retriever" "mixed breed" | Out-Null

    # Row 3: Color
    Set-Label 3 "Primary Color" "(dominant coat color)"
    Add-TextCC 3 "color" "e.g. black, brown, white" "brown" | Out-Null

    # Row 4: Age
    Set-Label 4 "Age (days)" "(365=1yr  730=2yr  1825=5yr)"
    Add-TextCC 4 "age_days" "e.g. 365" "365" | Out-Null

    # Row 5: Sex
    Set-Label 5 "Sex" ""
    Add-DropdownCC 5 "sex" @("Male","Female") "Male" | Out-Null

    # Row 6: Intake Condition
    Set-Label 6 "Intake Condition" ""
    Add-DropdownCC 6 "intake_condition" @("Normal","Sick","Injured","Aggressive") "Normal" | Out-Null

    # Row 7: Intake Type
    Set-Label 7 "Intake Type" ""
    Add-DropdownCC 7 "intake_type" @("Stray","Owner Surrender","Return") "Stray" | Out-Null

    # Alternate row shading
    for ($r = 2; $r -le 7; $r++) {
        $bg = if ($r % 2 -eq 0) { 0xF8F4F0 } else { 0xFFFFFF }
        $table.Cell($r,1).Shading.BackgroundPatternColor = $bg
        $table.Cell($r,2).Shading.BackgroundPatternColor = $bg
    }

    Write-Host "  Input form table built."

    # ── Move selection past the table ─────────────────────────────────────────
    $sel.EndKey($wdStory) | Out-Null
    $sel.TypeParagraph()
    $sel.TypeParagraph()

    # ── Run Query button (MACROBUTTON field) ──────────────────────────────────
    Write-Host "Adding Run Query button..."
    $sel.ParagraphFormat.Alignment = $wdAlignParagraphCenter

    $btnRange = $sel.Range
    $btnRange.Collapse($wdCollapseEnd)

    # Add the MACROBUTTON field
    $field = $doc.Fields.Add($btnRange, 55, "MACROBUTTON PredictDogLOS  Run Query", $true)
    # wdFieldMacroButton = 55

    # Style the button text
    $field.Result.Bold   = $true
    $field.Result.Font.Size  = 14
    $field.Result.Font.Color = 16777215       # white
    $field.Result.Font.Name  = "Calibri"
    $field.Result.Shading.BackgroundPatternColor = 0x3D2E1E  # dark teal (BGR)

    $sel.EndKey($wdStory) | Out-Null
    $sel.TypeParagraph()
    $sel.TypeParagraph()
    $sel.ParagraphFormat.Alignment = $wdAlignParagraphLeft

    Write-Host "  Run Query button added."

    # ── Results section ───────────────────────────────────────────────────────
    Write-Host "Adding results area..."
    $sel.Style = $doc.Styles("Heading 2")
    $sel.TypeText("5.2  Prediction Result")
    $sel.TypeParagraph()
    $sel.Style = $doc.Styles("Normal")
    $sel.TypeText("After clicking Run Query, the result will appear below:")
    $sel.TypeParagraph()

    $resRange = $sel.Range
    $resRange.Collapse($wdCollapseEnd)

    $resultCC = $doc.ContentControls.Add($wdContentControlRichText, $resRange)
    $resultCC.Tag          = "query_result"
    $resultCC.Title        = "query_result"
    $resultCC.LockContents = $false
    $resultCC.PlaceholderText.SetPlaceholderText("", $null, "(result will appear here after you click Run Query)")
    $resultCC.Range.Font.Name = "Courier New"
    $resultCC.Range.Font.Size = 10
    $resultCC.Range.Shading.BackgroundPatternColor = 0xF8F4F0

    Write-Host "  Results area added."

    # ── Save as .docm ─────────────────────────────────────────────────────────
    Write-Host "Saving as .docm..."
    $doc.SaveAs2($OutputPath, $wdFormatDocumentMacroEnabled)
    $doc.Close($false)

    Write-Host ""
    Write-Host "SUCCESS!" -ForegroundColor Green
    Write-Host "Output: $OutputPath"
    Write-Host ""
    Write-Host "How to use:"
    Write-Host "  1. Start the model server:  python app.py  (in a terminal)"
    Write-Host "  2. Open $OutputPath"
    Write-Host "  3. Enable macros when Word asks"
    Write-Host "  4. Fill in the form and click  Run Query"

} catch {
    Write-Error "Failed: $_"
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
