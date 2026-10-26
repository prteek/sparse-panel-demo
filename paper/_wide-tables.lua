-- PDF only: typeset tables with many columns at their natural width and shrink them to the
-- line width, so long unbreakable cells (codes, snake_case headers) never wrap or overlap.
-- Short, uncaptioned tables only; anything else is left to pandoc's default longtable.

local MIN_COLS = 8
local MAX_ROWS = 40

local function n_rows(tbl)
  local n = #tbl.head.rows
  for _, body in ipairs(tbl.bodies) do
    n = n + #body.body
  end
  return n
end

function Table(tbl)
  if not FORMAT:match("latex") then
    return nil
  end
  local has_caption = tbl.caption and tbl.caption.long and #tbl.caption.long > 0
  if #tbl.colspecs < MIN_COLS or n_rows(tbl) > MAX_ROWS or has_caption then
    return nil
  end

  -- natural column widths: no p{} columns, so cells never wrap
  for i, spec in ipairs(tbl.colspecs) do
    tbl.colspecs[i] = { spec[1], pandoc.ColWidthDefault }
  end

  local latex = pandoc.write(pandoc.Pandoc({ tbl }), "latex")
  latex = latex
    :gsub("\\begin{longtable}%[%]", "\\begin{tabular}")
    :gsub("\\endhead\n", "")
    :gsub("\\bottomrule\\noalign{}\n\\endlastfoot\n", "")
    :gsub("\\end{longtable}", "\\bottomrule\\noalign{}\n\\end{tabular}")

  return pandoc.RawBlock("latex",
    "\\begin{center}\\adjustbox{max width=\\linewidth}{%\n" .. latex .. "}\\end{center}")
end
