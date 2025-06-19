build_lib_example(
  NAME arpmec-example
  SOURCE_FILES arpmec-example.cc
  LIBRARIES_TO_LINK
    ${libwifi}
    ${libinternet}
    ${libarpmec}
    ${libinternet-apps}
)
