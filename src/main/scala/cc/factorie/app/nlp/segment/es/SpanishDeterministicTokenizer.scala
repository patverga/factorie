/* Copyright (C) 2008-2014 University of Massachusetts Amherst.
   This file is part of "FACTORIE" (Factor graphs, Imperative, Extensible)
   http://factorie.cs.umass.edu, http://github.com/factorie
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */
package cc.factorie.app.nlp.segment.es

import cc.factorie.app.nlp.{Document, DocumentAnnotator, Token}

/** Split a String into a sequence of Tokens.  Aims to adhere to tokenization rules used in Ontonotes and Penn Treebank.
    Note that CoNLL tokenization would use tokenizeAllDashedWords=true.
    Punctuation that ends a sentence should be placed alone in its own Token, hence this segmentation implicitly defines sentence segmentation also.
    (Although our the DeterministicSentenceSegmenter does make a few adjustments beyond this tokenizer.)
    @author Andrew McCallum
    */
class SpanishDeterministicTokenizer(caseSensitive:Boolean = false, tokenizeSgml:Boolean = false, tokenizeNewline:Boolean = false, tokenizeAllDashedWords:Boolean = false, abbrevPreceedsLowercase:Boolean = false) extends DocumentAnnotator {

  /** How the annotation of this DocumentAnnotator should be printed in one-word-per-line (OWPL) format.
      If there is no per-token annotation, return null.  Used in Document.owplString. */
  def tokenAnnotationString(token: Token) = token.stringStart.toString+'\t'+token.stringEnd.toString
  
  val patterns = new scala.collection.mutable.ArrayBuffer[String]

//  val html = "(?:<script[^>]*>(?:[^\u0000](?!</script>))*[^\u0000]?</script>)|(?:<style[^>]*>(?:[^\u0000](?!</style>))*[^\u0000]?</style>)"; if (!tokenizeSgml) patterns += html // The [^\u0000] ensures we match newline also
//  val htmlComment = "(?:<|&lt;)!--(?:[^\u0000](?!-->))*[^\u0000]?--(?:>|&gt;)"; patterns += htmlComment
//  val sgml2 = "<%(?:[^\u0000](?!%>))*[^\u0000]?%>"; patterns += sgml2 // Some HTML contains "<% blah %>" tags.
//  val sgml = "</?[A-Za-z!].*?(?<!%)>"; patterns += sgml // Closing with "%>" doesn't count
//  val htmlSymbol = "&(?:HT|TL|UR|LR|QC|QL|QR|amp|copy|reg|trade|odq|nbsp|cdq|lt|gt|#[0-9A-Za-z]+);"; patterns += htmlSymbol // TODO Make this list complete
//  val url = "https?://[^ \t\n\f\r\"<>|()]+[^ \t\n\f\r\"<>|.!?(){},-]"; patterns += url
//  val url2 = "(?:(?:www\\.(?:[^ \t\n\f\r\"<>|.!?(){},]+\\.)+[a-zA-Z]{2,4})|(?:(?:[^ \t\n\f\r\"`'<>|.!?(){},-_$]+\\.)+(?:com|org|net|edu|gov|cc|info|uk|de|fr|ca)))(?:/[^ \t\n\f\r\"<>|()]+[^ \t\n\f\r\"<>|.!?(){},-])?"; patterns += url2 // Common simple URL without the http
//  val url3 = "[A-Z]*[a-z0-9]+\\.(?:com|org|net|edu|gov|co\\.uk|ac\\.uk|de|fr|ca)"; patterns += url3 // Common, simple URL without the http or www.
//  val email = "(?:mailto:)?\\w+[-\\+\\.'\\w]*@(?:\\w+[-\\.\\+\\w]*\\.)*\\w+"; patterns += email
//  val usphone = "(?:\\+?1[-\\. \u00A0]?)?(?:\\(?:[0-9]{3}\\)[ \u00A0]?|[0-9]{3}[- \u00A0\\.])[0-9]{3}[\\- \u00A0\\.][0-9]{4}"; patterns += usphone
//  val frphone = "(?:\\+33)?(?:\\s[012345][-\\. ])?[0-9](?:[-\\. ][0-9]{2}){3}"; patterns += frphone
//  val date = "(?:(?:(?:(?:19|20)?[0-9]{2}[\\-/][0-3]?[0-9][\\-/][0-3]?[0-9])|(?:[0-3]?[0-9][\\-/][0-3]?[0-9][\\-/](?:19|20)?[0-9]{2}))(?![0-9]))"; patterns += date // e.g. 3/4/1992 or 2012-04-05, but don't match just the first 8 chars of 12-25-1112
//  val decade = "(?:19|20)?[0-9]0s"; patterns += decade
//  val currency = "(?:US|AU|NZ|C|CA|FJ|JY|HK|JM|KY|LR|NA|SB|SG|NT|BB|XC|BM|BN|BS|BZ|ZB|B)?\\$|&(?:euro|cent|pound);|\\p{Sc}|(?:USD|EUR|JPY|GBP|CHF|CAD|KPW|RMB|CNY|AD|GMT)(?![A-Z])"; patterns += currency
//  val hashtag = "#[A-Za-z][A-Za-z0-9]+"; patterns += hashtag // For Twitter
//  val atuser  = "@[A-Za-z][A-Za-z0-9]+"; patterns += atuser  // For Twitter
//  val emoticon = "[#<%\\*]?[:;!#\\$%@=\\|][-\\+\\*=o^<]{0,4}[\\(\\)oODPQX\\*3{}\\[\\]]{1,5}[#><\\)\\(]?(?!\\S)|'\\.'"; patterns += emoticon // Optional hat, eyes, optional repeated nose, mouth{1,5}, optional beard.  Or horizontal eyes '.'
//  val filename = "\\S+\\.(?:3gp|7z|ace|ai(?:f){0,2}|amr|asf|asp(?:x)?|asx|avi|bat|bin|bmp|bup|cab|cbr|cd(?:a|l|r)|chm|dat|divx|dll|dmg|doc|dss|dvf|dwg|eml|eps|exe|fl(?:a|v)|gif|gz|hqx|(?:s)?htm(?:l)?|ifo|indd|iso|jar|jsp|jp(?:e)?g|key|lnk|log|m4(?:a|b|p|v)|mcd|mdb|mid|mov|mp(?:2|3|4)|mp(?:e)?g|ms(?:i|wmm)|numbers|ogg|pages|pdf|php|png|pps|ppt|ps(?:d|t)?|Penn|pub|qb(?:b|w)|qxd|ra(?:m|r)|rm(?:vb)?|rtf|se(?:a|s)|sit(?:x)?|sql|ss|swf|tgz|tif|torrent|ttf|txt|vcd|vob|wav|wm(?:a|v)|wp(?:d|s)|xls|xml|xtm|zip)"; patterns += filename
//
//  // Abbreviation handling
//  val consonantNonAbbrevs = "(?:Ng|cwm|nth|pm)(?=\\.)"; patterns += consonantNonAbbrevs // the "val abbrev" below matches all sequences of consonants followed by a period; these are exceptions to that rule
//  val month = "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec" // Note that "May" is not included because it is not an abbreviation
//  val day = "Mon|Tue|Tues|Wed|Thu|Thurs|Fri"
//  val state = "Ala|Alab|Ariz|Ark|Calif|Colo|Conn|Del|Fla|Ill|Ind|Kans|Kan|Ken|Kent|Mass|Mich|Minn|Miss|Mont|Nebr|Neb|Nev|Dak|Okla|Oreg|Tenn|Tex|Virg|Wash|Wis|Wyo"
//  val state2 = "Ak|As|Az|Ar|Ca|Co|Ct|De|Fm|Fl|Ga|Gu|Hi|Id|Il|Ia|Ks|Ky|La|Mh|Md|Ma|Mi|Mn|Ms|Mo|Mt|Ne|Nv|Mp|Pw|Pa|Pr|Tn|Tx|Ut|Vt|Vi|Va|Wa|Wi|Wy"
//  // Removed two-word state abbreviations and also ME, OK, OR, OH
//  val honorific = "Adm|Attys?|Brig|Capts?|Cols?|Comms?|Co?mdrs?|Cpls?|Cpts?|Det|Drs?|Hon|Gens?|Govs?|Lieuts?|Lts?|Majs?|Miss|Messrs|Mr|Mrs|Ms|Pfc|Pres|Profs?|Pvts?|Reps?|Revs?|Sens?|Sgts?|Spc|Supts?"
//  val suffix = "Bros?|Esq|Jr|Ph\\.?[Dd]|Sr"
//  val place = "Aly|Anx|Ave?|Avn|Blvd|Boul|Byp|Cir|Cor|Rd|Squ|Sta|Ste?|Str|Ln|Mt"
//  val units = "in|fur|mi|lea|drc|oz|qtr|cwt" // not covered by "sequence of consonants" in abbrev below; note the abbrev pattern below insists on initial capital letter
//  val org = "Alt|Assns?|Bancorp|Bhd|Cos?|Comm|Comp|Corps?|Depts?|Elec|Inc|Inst|Intl|Lib|Ltd|M[ft]g|Mus|Natl|Plc|Pty|Sci|Ser|Sys|Univ"
//  val latin = "e.g|i.e" // i.e. and e.g.
//  val abbrev = "etc|vol|rev|dea|div|ests?|exp|exts?|gal|[BCDFGHJKLMNPQRSTVWX][bcdfghjklmnpqrstvwx]+"
//  val abbrevs = Seq(month, day, state, state2, honorific, suffix, place, units, org, latin, abbrev).flatMap(_.split('|').map(_.trim).filter(_.length > 0).map(_ + "\\.")).mkString("|")
//  patterns += abbrevs
//  val noAbbrev = "[Nn]o\\.(?=\\p{Z}*\\p{Nd})"; patterns += noAbbrev // Capture "No." when followed by a number (as in "No. 5", where it is an abbreviation of "number").  // TODO Consider a token normalization to "number"? -akm
//  val latin2 = "(?:i.e|e.g)(?!\\p{L})"; patterns += latin2 // i.e e.g
//
//  //val htmlLetter = "(?:&[aeiounyAEIOUNY](acute|grave|uml|circ|ring|tilde);)"
//  val htmlAccentedLetter = "(?:&[aeiouyntlAEIOUYNTL](?:acute|grave|uml|circ|orn|tilde|ring);)"; patterns += htmlAccentedLetter // TODO Make this list complete; see http://www.starr.net/is/type/htmlcodes.html
//  val letter = s"(?:[\\p{L}\\p{M}]|$htmlAccentedLetter)"
//  val ap = "(?:['\u0092\u2019]|&(?:apos|rsquo|#00?39|#00?92|#2019);)" // apostrophe and open single quotes
//  val ap2 = s"(?:$ap|&lsquo;|[`\u0091\u2018\u201B])" // also includes backwards apostrophe and single quotes, but not double quotes
//  //val enumeration = ""; patterns += enumeration // a.1, 4,5, 6:7, 8-9, 0/1, '2, 3's, 3'4, 5'b, a'6, a'b // TODO Do we really want something like this?
//  val contraction2 = "what(?=cha)|wan(?=na)"; patterns += contraction2 // contractions without apostrophes
//  val contraction = s"(?:[nN]${ap}[tT]|(?<=\\p{L})$ap(?:d|D|s|S|m|M|re|RE|ve|VE|ll|LL)(?!\\p{L}))"; patterns += contraction // an apostrophe, preceded by a non-consumed letter, followed by patterns of contractions
//  val apword = s"${ap}nt|${ap}n(?:$ap)?|${ap2}em|[OoDdLl]${ap}$letter+|[Oo]${ap2}clock|ma${ap2}am|add${ap2}l|[Cc]${ap2}mon|${ap2}cause|${ap}till?|ol$ap|Dunkin$ap|$ap[1-9]0s|N$ap|\\p{L}\\p{Ll}*[aeiou]$ap[aeiou]\\p{Ll}*"; patterns += apword // words that include an apostrophe, like O'Reilly, C'mon, 'n', Shi'ite, 20's, N'goma
//  //val ing = s"[A-Za-z]{3,}in${ap}"; patterns += ing // fishin' (but this pattern also gets all but the last character of "Britain's" :-(  // TODO Try to find some more specific fix for this
//  val initials2 = "\\p{L}(?:\\.\\p{L})+(?!\\p{P})"; patterns += initials2 // A.B  This must come before 'initials' or else 'initials' will capture the prefix.
//  val initials = "(?:\\p{L}\\.)+(?![\\.!\\?]{2}|\\.\\p{L})"; patterns += initials // A.  A.A.A.I.  and U.S. in "U.S..", etc., but not A... or A..B
//  //val briefAbbrevs = "[A-Z][a-z]?\\."; patterns += briefAbbrevs // and initials; so includes A. and Mr. but not Mrs. Calif. or Institute.  Removed because otherwise we get "me." and "it."
//  val ordinals = "[0-9]{1,4}(?:st|nd|rd|th)"; patterns += ordinals // like 1st and 22nd
//  val quote = "''|``|[\u2018\u2019\u201A\u201B\u201C\u201D\u0091\u0092\u0093\u0094\u201A\u201E\u201F\u2039\u203A\u00AB\u00BB]{1,2}|[\"\u201C\u201D\\p{Pf}]|&(?:quot|[rl][ad]quo);|" + ap2 + "{2}"; patterns += quote
//  if (tokenizeAllDashedWords) { val dashedWord = s"(?:$letter)(?:[\\p{L}\\p{M}\\p{Nd}_]*(?:-[\\p{L}\\p{M}\\p{Nd}_]*)*)"; patterns += dashedWord }
//  // List of prefixes taken from http://en.wikipedia.org/wiki/English_prefixes with the addition of "e", "uh" and "x" from Ontonotes examples.
//  val dashedPrefixes = "(?i:a|anti|arch|be|co|counter|cross|de|dis|e|en|em|ex|fore|hi|hind|mal|mid|midi|mini|mis|o|out|over|part|post|pre|pro|re|self|step|t|trans|twi|un|under|up|with|Afro|ambi|amphi|an|ana|Anglo|ante|apo|astro|auto|bi|bio|circum|cis|con|com|col|cor|contra|cryo|crypto|de|demi|demo|deutero|deuter|di|dia|dis|dif|du|duo|eco|electro|e|en|epi|Euro|ex|extra|fin|Franco|geo|gyro|hetero|hemi|homo|hydro|hyper|hypo|ideo|idio|in|Indo|in|infra|inter|intra|iso|macro|maxi|mega|meta|micro|mono|multi|neo|non|omni|ortho|paleo|pan|para|ped|per|peri|photo|pod|poly|post|pre|preter|pro|pros|proto|pseudo|pyro|quasi|retro|semi|socio|sub|sup|super|supra|sur|syn|tele|trans|tri|uh|ultra|uni|vice|x)"
//  val dashedSuffixes = "(?i:able|ahol|aholic|ation|centric|cracy|crat|dom|e-\\p{L}+|er|ery|esque|ette|fest|fi|fold|ful|gate|gon|hood|ian|ible|ing|isation|ise|ising|ism|ist|itis|ization|ize|izing|less|logist|logy|ly|most|o-torium|rama|ise)"
//  val dashedPrefixWord = dashedPrefixes+"-[\\p{L}\\p{M}][\\p{L}\\p{M}\\p{Nd}]*"; patterns += dashedPrefixWord // Dashed words with certain prefixes, like "trans-ocean" or "Afro-pop"
//  val dashedSuffixWord = "[\\p{L}\\p{M}\\p{N}]+-"+dashedSuffixes+s"(?!$letter)"; patterns += dashedSuffixWord // Dashed words with certain suffixes, like "senior-itis" // TODO Consider a dashedPrefixSuffixWord?
//  // common dashed words in Ontonotes include counter-, ultra-, eastern-, quasi-, trans-,...
//  val fraction = "[\u00BC\u00BD\u00BE\u2153\u2154]|&(?:frac14|frac12|frac34);|(?:\\p{N}{1,4}[- \u00A0])?\\p{N}{1,4}(?:\\\\?/|\u2044)\\p{N}{1,4}"; patterns += fraction
//  val contractedWord = s"[\\p{L}\\p{M}]+(?=(?:$contraction))"; patterns += contractedWord // Includes any combination of letters and accent characters before a contraction
//  val caps = s"\\p{Lu}+(?:[&+](?!(?:$htmlSymbol|$htmlAccentedLetter))(?:\\p{Lu}(?!\\p{Ll}))+)+"; patterns += caps // For "AT&T" but don't grab "LAT&Eacute;" and be sure not to grab "PE&gym"
//  val word = s"$letter(?:[\\p{L}\\p{M}\\p{Nd}_]|$letter)*+"; patterns += word // Includes any combination of letters, accent characters, numbers and underscores, dash-followed-by-numbers (as in "G-20" but not "NYT-03-04-2012").  It may include a & as long as it is followed by a letter but not an HTML symbol encoding
//
//  // TODO Not sure why the pattern above is not getting the last character of a word ending in \u00e9 -akm
//  val number = s"(?<![\\p{Nd}])[-\\+\\.,]?(?!$date)\\p{Nd}+(?:[\\.:,]\\p{Nd}+)*"; patterns += number // begin with an optional [+-.,] and a number, followed by numbers or .:, punc, ending in number.  Avoid matching dates inside "NYT-03-04-2012".  Cannot be preceded by number (or letter? why?  do we need "USD32"?), in order to separate "1989-1990" into three tokens.
//  val number2 = ap+"\\p{Nd}{2}"; patterns += number2 // For years, like '91
//  patterns += ap2 // Defined earlier for embedded use, but don't include in patterns until here
//  val ellipsis = "\\.{2,5}(?![!\\?])|(?:\\.[ \u00A0]){2,4}\\.|[\u0085\u2026]"; patterns += ellipsis // catch the ellipsis not captured in repeatedPunc, such as ". . ." and unicode ellipsis.  Include \\.{2,5} for use in TokenNormalizer1  Don't capture "...?!"; let repeatedPunc do that.
//  val repeatedPunc = "[,~\\*=\\+\\.\\?!#]+|-{4,}"; patterns += repeatedPunc // probably used as ASCII art
//  val mdash = "-{2,3}|&(?:mdash|MD);|[\u2014\u2015]"; patterns += mdash
//  val dash = "&(?:ndash);|[-\u0096\u0097\\p{Pd}]"; patterns += dash // I think \p{Pd} should include \u2013\u2014\u2015
//  val punc = "\\p{P}"; patterns += punc // This matches any kind of punctuation as a single character, so any special handling of multiple punc together must be above, e.g. ``, ---
//  val symbol = "\\p{S}|&(?:degree|plusmn|times|divide|infin);"; patterns += symbol
//  val htmlChar = "&[a-z]{3,6};"; patterns += htmlChar // Catch-all, after the more specific cases above, including htmlSymbol
//  val catchAll = "\\S"; patterns += catchAll // Any non-space character.  Sometimes, due to contextual restrictions above, some printed characters can slip through.  It will probably be an error, but at least the users will see them with this pattern.
//  val newline = "\n"; if (tokenizeNewline) patterns += newline
//  val space = "(?:\\p{Z}|&nbsp;)+" // but not tokenized here

  // referenced but not added to patters
  val HASHES = s"#+"
  val APOS = s"['\u0092\u2019]|&apos;"
  val APOSETCETERA = s"$APOS|[\u0091\u2018\u201B]"
  val SPLET = s"&[aeiouAEIOU](acute|grave|uml)"
  val NEWLINE = s"\r|\r?\n|\u2028|\u2029|\u000B|\u000C|\u0085"
  /* Handles Nko numerals */
  val DIGIT = s"[0-9]|[\u07C0-\u07C9]"
  /* For some reason U+0237-U+024F (dotless j) isn't in [:letter:]. Recent additions? */
  val CHAR = s"\\p{IsLatin}|$SPLET|[\u00AD\u0237-\u024F\u02C2-\u02C5\u02D2-\u02DF\u02E5-\u02FF\u0300-\u036F\u0370-\u037D\u0384\u0385\u03CF\u03F6\u03FC-\u03FF\u0483-\u0487\u04CF\u04F6-\u04FF\u0510-\u0525\u055A-\u055F\u0591-\u05BD\u05BF\u05C1\u05C2\u05C4\u05C5\u05C7\u0615-\u061A\u063B-\u063F\u064B-\u065E\u0670\u06D6-\u06EF\u06FA-\u06FF\u070F\u0711\u0730-\u074F\u0750-\u077F\u07A6-\u07B1\u07CA-\u07F5\u07FA\u0900-\u0903\u093C\u093E-\u094E\u0951-\u0955\u0962-\u0963\u0981-\u0983\u09BC-\u09C4\u09C7\u09C8\u09CB-\u09CD\u09D7\u09E2\u09E3\u0A01-\u0A03\u0A3C\u0A3E-\u0A4F\u0A81-\u0A83\u0ABC-\u0ACF\u0B82\u0BBE-\u0BC2\u0BC6-\u0BC8\u0BCA-\u0BCD\u0C01-\u0C03\u0C3E-\u0C56\u0D3E-\u0D44\u0D46-\u0D48\u0E30-\u0E3A\u0E47-\u0E4E\u0EB1-\u0EBC\u0EC8-\u0ECD]"
  val HYPHEN = s"[-_\u058A\u2010\u2011]"
  /* prefixed compounds that shouldn't be split off */
  val PREFIX = s"(anti|co|ex|meso|neo|pre|pro|quasi|re|semi|sub)$HYPHEN"


  val NUM = s"$DIGIT+|$DIGIT*([-_.:,\u00AD\u066B\u066C]$DIGIT+)+"
  val WORD = s"($CHAR)+"
  val WORD2 = s"$WORD(($NUM|$APOSETCETERA)$WORD?)+$WORD"
  val WORD3 = s"$NUM?($WORD|$WORD2)$NUM?"
  /* all types of "words" (word, word2, word3) */
  val ANYWORD = s"(($WORD)|($WORD2)|($WORD3))"
  
  /* Patterns to help identify the various types of verb+enclitics */
  val OS = s"os(l[oa]s?)?"
  val ATTACHED_PRON = s"((me|te|se|nos|les?)(l[oa]s?)?)|l[oa]s?"
  val VB_IRREG = s"d[ií]|h[aá]z|v[eé]|p[oó]n|s[aá]l|sé|t[eé]n|v[eé]n"
  val VB_REG = s"$WORD([aeiáéí]r|[áé]ndo|[aeáé]n?|[aeáé]mos?)"
  val VB_PREF = s"$VB_IRREG|($VB_REG)"

  /* ABBREVIATIONS - INDUCED FROM 1987 WSJ BY HAND */
  val ABMONTH = s"ene|feb|mar|abr|may|jun|jul|ago|sep|set|sept|oct|nov|dic"
  val ABDAYS = s"lun|mar|mi[\u00E9e]|jue|vie|sab|dom"
  val ABTITLE = s"Mr|Mrs|Ms|[M]iss|Drs?|Profs?|Sens?|Reps?|Attys?|Lt|Col|Gen|Messrs|Govs?|Adm|Rev|Maj|Sgt|Cpl|Pvt|Mt|Capt|Ste?|Ave|Pres|Lieut|Hon|Brig|Co?mdr|Pfc|Spc|Supts?|Det|M|MM|Mme|Mmes|Mlle|Mlles"
  val ABTITLE_ES = s"Sr|Sra|Srta|D|Da|D[\u00F1n]a|Dr|Dra|Prof|Profa|Gob|Gral"
  val ABPTIT_ES = s"Av|Avda|apdo|Esq|Uds?|Vds?"
  /* Bhd is Malaysian companies! */
  /* TODO: Change the class of at least Pty as usually another one like Ltd following... */
  val ABCOMP = s"Inc|Cos?|Corp|Pp?tys?|Ltd|Plc|Bancorp|Dept|Bhd|Assn|Univ|Intl|Sys"
  val ABCOMP2 = s"Invt|Elec|Natl|M[ft]g"
 
  val ATS = s"@+"
  val UNDS = s"_+"
  
  
  
  // actual patterns
  /* Don't allow SGML to cross lines, even though it can...
   Really SGML shouldn't be here at all, it's kind of legacy. */
  val SGML = s"<\\/?[A-Za-z!?][^>\r\n]*>"; patterns += SGML
    val SPMDASH = s"&(MD|mdash|ndash);|[\u0096\u0097\u2013\u2014\u2015]"; patterns += SPMDASH
  /* Spanish ordinals */
  val ORDINAL = s"[0-9]*([13].?er|[0-9].?[oa\u00BA\u00AA\u00B0])" //; patterns += ORDINAL
  /* \u3000 is ideographic space */
  val SPACE = s"[ \t\u00A0\u2000-\u200A\u3000]"; patterns += s"$ORDINAL(?=$SPACE)"

  val SPAMP = s"&amp;"; patterns += SPAMP
  val SPPUNC = s"&(HT|TL|UR|LR|QC|QL|QR|odq|cdq|#[0-9]+)"; patterns += SPPUNC

  /* European 24hr time expression e.g. 20h14 */
  val TIMEXP = s"$DIGIT{1,2}(h)$DIGIT{1,2}"; patterns += TIMEXP

  /* Spanish contractions:
*
* al => a + l (to the)
* del => de + l (of the)
* conmigo/contigo/consigo => con + migo/tigo/sigo (with me/you/them)
*
*/
  val CONTRACTION = s"del|al|con[mts]igo"; patterns += CONTRACTION

  /* Handles all other verbs with attached pronouns  */
  val VB_ATTACHED_PRON = s"($VB_PREF)$ATTACHED_PRON|$OS"; patterns += VB_ATTACHED_PRON
   /* Handles second person plural imperatives:
  *
  * Sentaos => Senta + os (seat + yourselves)
  * Vestíos => Vestí + os (dress + yourselves)
  */
  val VB_2PP_PRON = s"($CHAR*)[aeiáéí]((d$ATTACHED_PRON)|$OS)"; patterns += VB_2PP_PRON
  val COMPOUND_NOSPLIT = s"$PREFIX$ANYWORD"; patterns += COMPOUND_NOSPLIT

  /* spanish compounds */
  val COMPOUND = s"$WORD($HYPHEN$WORD)+"; patterns += COMPOUND
  /* Handles Arabic numerals & soft-hyphens */
//  patterns += NUM
  /* common units abbreviated - to differentiate between WORD_NUM
  * as WORD_NUMs shouldn't be split but these should. */
  val UNIT_PREF = "[dcm\u00B5\u03BCnpfazyhkMGTPEZY]|da"; //patterns += UNIT_PREF
  val UNIT = s"($UNIT_PREF)?m|kg|s|[A]|[K]|mol|rad|[H]z|N|Pa|J|W|C|V"; patterns +=s"$NUM(?=$UNIT)"

  /* Includes words with numbers, eg. sp3 */
  patterns += WORD2
  /* Includes words with apostrophes in the middle (french, english, catalan loanwords) */
  patterns += WORD3
  patterns += WORD


  /* URLs, email, and Twitter handles
   Technically, Twitter names should be capped at 15 characters.
   However, then you get into weirdness with what happens to the
   rest of the characters. */
  val FULLURL = "https?:\\/\\/[^ \t\n\f\r\"<>|()]+[^ \t\n\f\r\"<>|.!?(){},-]"; patterns += FULLURL
  val LIKELYURL = "((www\\.([^ \t\n\f\r\"<>|.!?(){},]+\\.)+[a-zA-Z]{2,4})|(([^ \t\n\f\r\"`'<>|.!?(){},-_\\$]+\\.)+(com|net|org|edu)))(\\/[^ \t\n\f\r\"<>|()]+[^ \t\n\f\r\"<>|.!?(){},-])?"; patterns += LIKELYURL
  val EMAIL = "[a-zA-Z0-9][^ \t\n\f\r\"<>|()\u00A0]*@([^ \t\n\f\r\"<>|().\u00A0]+\\.)*([^ \t\n\f\r\"<>|().\u00A0]+)"; patterns += EMAIL
  val TWITTER_NAME = s"@[a-zA-Z_][a-zA-Z_0-9]*"; //patterns += TWITTER_NAME
  val TWITTER_CATEGORY = s"#$WORD"; //patterns += TWITTER_CATEGORY
  val TWITTER = s"$TWITTER_NAME|$TWITTER_CATEGORY"; patterns += TWITTER

  val DATE = s"$DIGIT{1,2}[\\-\\/]$DIGIT{1,2}[\\-\\/]$DIGIT{2,4}"; patterns += DATE
  /* Now don't allow bracketed negative numbers!
     They have too many uses (e.g., years or times in parentheses), and
     having them in tokens messes up treebank parsing. */
  val NUMBER = s"[\\-+]?$NUM"; patterns += NUMBER
  val SUBSUPNUM = s"[\u207A\u207B\u208A\u208B]?([\u2070\u00B9\u00B2\u00B3\u2074-\u2079]+|[\u2080-\u2089]+)"; patterns += SUBSUPNUM
  /* These are cent and pound sign, euro and euro, and Yen, Lira */
  val MONEYSIGN = "[\\$\u00A2\u00A3\u00A4\u00A5\u0080\u20A0\u20AC\u060B\u0E3F\u20A4\uFFE0\uFFE1\uFFE5\uFFE6]"; patterns += MONEYSIGN
  /* Constrain fraction to only match likely fractions */
  val FRAC = s"($DIGIT{1,4}[- \u00A0])?$DIGIT{1,4}(\\?\\/|\u2044)$DIGIT{1,4}"; patterns += FRAC
  val FRACSTB3 = s"($DIGIT{1,4}-)?$DIGIT{1,4}(\\?\\/|\u2044)$DIGIT{1,4}"; patterns += FRACSTB3
  val FRAC2 = s"[\u00BC\u00BD\u00BE\u2153-\u215E]"; patterns += FRAC2

  /* ABBREV1 abbreviations are normally followed by lower case words.  If
     they're followed by an uppercase one, we assume there is also a
     sentence boundary */
  val ABBREV1 = s"($ABMONTH|$ABDAYS|$ABCOMP|$ABPTIT_ES|etc|al|seq|p\\.ej)\\." //; patterns += ABBREV1
  val SPACENL = s"($SPACE|$NEWLINE)"
  val SENTEND = s"$SPACENL($SPACENL|([A-Z]|$SGML))"; patterns += s"$ABBREV1(?=($SENTEND))"
  /* ABRREV2 abbreviations are normally followed by an upper case word.  We
     assume they aren't used sentence finally */
  /* ACRO Is a bad case -- can go either way! */
  val ACRO = s"[A-Za-z]{1,2}(\\.[A-Za-z]{1,2})+|(Canada|Sino|Korean|EU|Japan|non)-U\\.S|U\\.S\\.-(U\\.K|U\\.S\\.S\\.R)"
  val ABBREV4 = s"[A-Za-z]|$ABTITLE|$ABTITLE_ES|vs|Alex|Wm|Jos|Cie|a\\.k\\.a|cf|TREAS|$ACRO|$ABCOMP2"
  val ABBREV2 = s"$ABBREV4\\."; patterns += ABBREV2
  patterns += s"$ABBREV4(?=$SPACE)"

  /* In the caseless world S.p.A. "Società Per Azioni (Italian: shared company)" is got as a regular acronym */
  /* ??? */
  patterns += s"$ACRO(?=$SPACENL)"
  val DBLQUOT = "\"|&quot;"; patterns += DBLQUOT
  val QUOTES = s"$APOSETCETERA|''|[`\u2018\u2019\u201A\u201B\u201C\u201D\u0091\u0092\u0093\u0094\u201E\u201F\u2039\u203A\u00AB\u00BB]{1,2}"; patterns += QUOTES
  /* phone numbers. keep multi dots pattern separate, so not confused with decimal numbers. */
  val PHONE = s"(\\([0-9]{2,3}\\)[ \u00A0]?|(\\+\\+?)?([0-9]{2,4}[\\- \u00A0])?[0-9]{2,4}[\\- \u00A0])[0-9]{3,4}[\\- \u00A0]?[0-9]{3,5}|((\\+\\+?)?[0-9]{2,4}\\.)?[0-9]{2,4}\\.[0-9]{3,4}\\.[0-9]{3,5}"; patterns += PHONE
  val OPBRAC = s"[<\\[{(]|&lt;"; patterns += OPBRAC
  val CLBRAC = s"[>\\])}]|&gt;"; patterns += CLBRAC
  val HYPHENS = s"\\-+"; patterns += HYPHENS
  val LDOTS = s"\\.{3,5}|(\\.[ \u00A0]){2,4}\\.|[\u0085\u2026]"; patterns += LDOTS

  val FNMARKS = s"$ATS|$HASHES|$UNDS"; patterns += FNMARKS
  val ASTS = s"\\*+|(\\\\*){1,3}"; patterns += ASTS
  val INSENTP = s"[,;:\u3001]"; patterns += INSENTP

  val ENDPUNCT = "[?!]+"; patterns += ENDPUNCT
  val STARTPUNCT = "[.¡¿\\u037E\\u0589\\u061F\\u06D4\\u0700-\\u0702\\u07FA\\u3002]"; patterns += STARTPUNCT
  patterns += "="

  /* Fake duck feet appear sometimes in WSJ, and aren't likely to be SGML, less than, etc., so group. */
  val FAKEDUCKFEET = s"<<|>>"; patterns += FAKEDUCKFEET
  /* U+2200-U+2BFF has a lot of the various mathematical, etc. symbol ranges */
  val MISCSYMBOL = s"[+%&~\\^|\\¦\u00A7¨\u00A9\u00AC\u00AE¯\u00B0-\u00B3\u00B4-\u00BA\u00D7\u00F7\u0387\u05BE\u05C0\u05C3\u05C6\u05F3\u05F4\u0600-\u0603\u0606-\u060A\u060C\u0614\u061B\u061E\u066A\u066D\u0703-\u070D\u07F6\u07F7\u07F8\u0964\u0965\u0E4F\u1FBD\u2016\u2017\u2020-\u2023\u2030-\u2038\u203B\u203E-\u2042\u2044\u207A-\u207F\u208A-\u208E\u2100-\u214F\u2190-\u21FF\u2200-\u2BFF\u3012\u30FB\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\uFF65]"; patterns += MISCSYMBOL
  val SPACES = s"$SPACE+"; patterns += SPACES













/* \uFF65 is Halfwidth katakana middle dot; \u30FB is Katakana middle dot */
/* Math and other symbols that stand alone: °²× ∀ */
// Consider this list of bullet chars: 2219, 00b7, 2022, 2024

val tokenRegexString = patterns.filter(_.length != 0).mkString("(",")|(",")")
//  val tokenRegex = if (!caseSensitive) ("(?i)"+tokenRegexString).r else tokenRegexString.r
  val tokenRegex = if (!caseSensitive) ("(?i)"+tokenRegexString).r else tokenRegexString.r

  def process(document: Document): Document = {
    for (section <- document.sections) {
      var prevTokenPeriod = false // Does the previous Token.string consist entirely of .
      val tokenIterator = tokenRegex.findAllIn(section.string)
      while (tokenIterator.hasNext) {
        tokenIterator.next()
        val string = document.string.substring(section.stringStart + tokenIterator.start, section.stringStart + tokenIterator.end)
        if (string.trim != "") {
          if (abbrevPreceedsLowercase && prevTokenPeriod && java.lang.Character.isLowerCase(string(0)) && section.length > 1 && section.tokens(section.length - 2).stringEnd == section.tokens(section.length - 1).stringStart) {
            // If we have a pattern like "Abbrev. has" (where "has" is any lowercase word) with token strings "Abbrev", ".", "is" (currently looking at "is")
            // then assume that the previous-previous word is actually an abbreviation; patch it up to become "Abbrev.", "has".
            val lastTwoTokens = section.takeRight(2).toIndexedSeq
            section.remove(section.length - 1);
            section.remove(section.length - 1)
            new Token(section, lastTwoTokens(0).stringStart, lastTwoTokens(1).stringEnd)
            new Token(section, section.stringStart + tokenIterator.start, section.stringStart + tokenIterator.end)
          } else if (tokenizeNewline && string == "\n") {
            new Token(section, section.stringStart + tokenIterator.start, section.stringStart + tokenIterator.end)
          } else if (tokenizeSgml ||
            !((string(0) == '<' && string(string.length - 1) == '>') // We have an SGML tag
              || (string(0) == '&' && string(string.length - 1) == ';') // We have an odd escaped SGML tag &gt;...&lt;
              || string.toLowerCase == "&nbsp;" // Don't make token from space
              )
          ) {
            new Token(section, section.stringStart + tokenIterator.start, section.stringStart + tokenIterator.end)
          }
          if (string == ".") prevTokenPeriod = true else prevTokenPeriod = false
        }
      }
    }
    if (!document.annotators.contains(classOf[Token]))
      document.annotators(classOf[Token]) = this.getClass
    document
  }
  def prereqAttrs: Iterable[Class[_]] = Nil
  def postAttrs: Iterable[Class[_]] = List(classOf[Token])
  
  /** Convenience function to run the tokenizer on an arbitrary String.  The implementation builds a Document internally, then maps to token strings. */
  def apply(s:String): Seq[String] = process(new Document(s)).tokens.toSeq.map(_.string)
}

object SpanishDeterministicTokenizer extends SpanishDeterministicTokenizer(false, false, false, false, false) {
  def main(args: Array[String]): Unit = {
    val string = io.Source.fromInputStream(System.in).mkString
    val doc = new Document(string)
    SpanishDeterministicTokenizer.process(doc)
    println(doc.tokens.map(_.string).mkString("\n"))
  }
}
